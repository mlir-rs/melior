//! Lowering of the remaining Toy, affine, arith, func, memref, and scf
//! operations to the LLVM dialect.

use crate::{
    error::Error,
    utility::{block_operations, check_lowering, is_operation, symbol_name},
};
use melior::{
    Context, GreedyRewriteDriverConfig, PatternRewriter, RewritePatternSet, RewriterBase,
    apply_patterns_and_fold_greedily, create_op_rewrite_pattern,
    dialect::{
        llvm::{
            attributes::{Linkage, linkage},
            r#type as llvm_type,
        },
        ods::{arith, llvm, memref, scf},
    },
    ir::{
        Attribute, Block, BlockLike, BlockRef, Location, Module, Operation, OperationRef, Region,
        RegionLike, Type, Value, ValueLike,
        attribute::{
            DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::{OperationLike, OperationMutLike, WalkOrder, WalkResult},
        r#type::{DimSize, IntegerType, MemRefType, ShapedTypeLike},
    },
    pass::{PassManager, conversion},
};

/// Lowers a module to the LLVM dialect.
pub fn lower_to_llvm<'c>(context: &'c Context, module: &mut Module<'c>) -> Result<(), Error> {
    let patterns = RewritePatternSet::new(context);

    patterns.add(create_op_rewrite_pattern(
        "toy.print",
        1,
        context,
        lower_print,
        &[
            "arith.constant",
            "llvm.call",
            "llvm.func",
            "llvm.mlir.addressof",
            "llvm.mlir.global",
            "memref.load",
            "scf.for",
            "scf.yield",
        ],
    ));

    apply_patterns_and_fold_greedily(
        module,
        &patterns.freeze(),
        &GreedyRewriteDriverConfig::new(),
    )?;

    check_lowering(module, |_| false)?;
    emit_c_interface(context, module);

    let pass_manager = PassManager::new(context);

    pass_manager.add_pass(conversion::create_lower_affine());
    pass_manager.add_pass(conversion::create_scf_to_control_flow());
    pass_manager.add_pass(conversion::create_arith_to_llvm());
    pass_manager.add_pass(conversion::create_finalize_mem_ref_to_llvm());
    pass_manager.add_pass(conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(conversion::create_func_to_llvm());
    pass_manager.add_pass(conversion::create_reconcile_unrealized_casts());

    pass_manager.run(module)?;

    Ok(())
}

// The execution engine invokes functions through their C interfaces.
fn emit_c_interface<'c>(context: &'c Context, module: &mut Module<'c>) {
    module
        .as_operation_mut()
        .walk_mut(WalkOrder::PreOrder, |mut operation| {
            if is_operation(&*operation, "func.func") && symbol_name(&*operation) == Some("main") {
                operation.set_attribute("llvm.emit_c_interface", Attribute::unit(context));

                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        });
}

// Lowers a print operation to a loop nest calling `printf` on each element of
// a memory reference.
fn lower_print<'c>(operation: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    // SAFETY: The context outlives the pattern application.
    let context = unsafe { rewriter.context().to_ref() };
    let Some((input, memref_type)) = operation
        .operand(0)
        .ok()
        .and_then(|input| Some((input, MemRefType::try_from(input.r#type()).ok()?)))
    else {
        return false;
    };
    let Some(shape) = (0..memref_type.rank())
        .map(|index| match memref_type.dim_size(index) {
            Ok(DimSize::Static(size)) => Some(size),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(module_body) = module_body(operation) else {
        return false;
    };
    let location = operation.location();

    declare_printf(context, &rewriter, module_body, location);

    let (Some(format_specifier), Some(newline)) = (
        global_string(
            context,
            &rewriter,
            module_body,
            operation,
            "format_specifier",
            "%f \0",
        ),
        global_string(
            context,
            &rewriter,
            module_body,
            operation,
            "newline",
            "\n\0",
        ),
    ) else {
        return false;
    };

    let mut induction_variables = vec![];

    for (dimension, &size) in shape.iter().enumerate() {
        let (Some(lower_bound), Some(upper_bound), Some(step), Ok(block)) = (
            index_constant(context, &rewriter, 0, location),
            index_constant(context, &rewriter, size as i64, location),
            index_constant(context, &rewriter, 1, location),
            Block::new(&[Type::index(context)], &[location]),
        ) else {
            return false;
        };

        // A newline is printed after each row of the innermost dimension.
        if dimension + 1 != shape.len() {
            block.append_operation(call_printf(context, &[newline], location));
        }

        block.append_operation(scf::r#yield(context, &[], location).into());

        let region = Region::new();
        region.append_block(block);

        let Some(body) = rewriter
            .insert(
                scf::r#for(
                    context,
                    &[],
                    lower_bound,
                    upper_bound,
                    step,
                    &[],
                    region,
                    location,
                )
                .into(),
            )
            .region(0)
            .ok()
            .and_then(|region| region.first_block())
        else {
            return false;
        };
        let Ok(induction_variable) = body.argument(0) else {
            return false;
        };

        induction_variables.push(induction_variable.into());
        rewriter.set_insertion_point_to_start(body);
    }

    let Ok(element) = rewriter
        .insert(
            memref::load(
                context,
                memref_type.element(),
                input,
                &induction_variables,
                location,
            )
            .into(),
        )
        .result(0)
    else {
        return false;
    };

    rewriter.insert(call_printf(
        context,
        &[format_specifier, element.into()],
        location,
    ));
    rewriter.erase_op(operation);

    true
}

fn module_body<'c, 'a>(operation: OperationRef<'c, 'a>) -> Option<BlockRef<'c, 'a>> {
    let mut operation = operation;

    while !is_operation(operation, "builtin.module") {
        operation = operation.parent_operation()?;
    }

    operation.region(0).ok()?.first_block()
}

fn index_constant<'c, 'r>(
    context: &'c Context,
    rewriter: &'r RewriterBase<'c, '_>,
    value: i64,
    location: Location<'c>,
) -> Option<Value<'c, 'r>> {
    let index_type = Type::index(context);

    rewriter
        .insert(
            arith::constant(
                context,
                index_type,
                IntegerAttribute::new(index_type, value).into(),
                location,
            )
            .into(),
        )
        .result(0)
        .ok()
        .map(Into::into)
}

// The signature of `printf` is `i32 (ptr, ...)`.
fn printf_type(context: &Context) -> Type<'_> {
    llvm_type::function(
        IntegerType::new(context, 32).into(),
        &[llvm_type::pointer(context, 0)],
        true,
    )
}

fn declare_printf<'c>(
    context: &'c Context,
    rewriter: &RewriterBase<'c, '_>,
    module_body: BlockRef<'c, '_>,
    location: Location<'c>,
) {
    if block_operations(module_body).any(|operation| {
        is_operation(operation, "llvm.func") && symbol_name(operation) == Some("printf")
    }) {
        return;
    }

    rewriter.set_insertion_point_to_start(module_body);
    rewriter.insert(
        llvm::func(
            context,
            Region::new(),
            StringAttribute::new(context, "printf"),
            TypeAttribute::new(printf_type(context)),
            location,
        )
        .into(),
    );
}

fn call_printf<'c>(
    context: &'c Context,
    arguments: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    llvm::CallOperation::builder(context, location)
        .result(Some(IntegerType::new(context, 32).into()))
        .callee_operands(arguments)
        .op_bundle_operands(&[])
        .op_bundle_sizes(DenseI32ArrayAttribute::new(context, &[]))
        .callee(Some(FlatSymbolRefAttribute::new(context, "printf")))
        .var_callee_type(Some(TypeAttribute::new(printf_type(context))))
        .build()
        .into()
}

// Returns a pointer to a global string, creating the string if necessary.
fn global_string<'c, 'r>(
    context: &'c Context,
    rewriter: &'r RewriterBase<'c, '_>,
    module_body: BlockRef<'c, '_>,
    operation: OperationRef<'c, '_>,
    name: &str,
    value: &str,
) -> Option<Value<'c, 'r>> {
    let location = operation.location();

    if !block_operations(module_body).any(|operation| {
        is_operation(operation, "llvm.mlir.global") && symbol_name(operation) == Some(name)
    }) {
        rewriter.set_insertion_point_to_start(module_body);
        rewriter.insert(
            llvm::GlobalOperation::builder(context, location)
                .initializer(Region::new())
                .global_type(TypeAttribute::new(llvm_type::array(
                    IntegerType::new(context, 8).into(),
                    value.len() as u32,
                )))
                .sym_name(StringAttribute::new(context, name))
                .linkage(linkage(context, Linkage::Internal))
                .constant(Some(Attribute::unit(context)))
                .value(Some(StringAttribute::new(context, value).into()))
                .build()
                .into(),
        );
    }

    rewriter.set_insertion_point_before(operation);

    rewriter
        .insert(
            llvm::mlir_addressof(
                context,
                llvm_type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, name),
                location,
            )
            .into(),
        )
        .result(0)
        .ok()
        .map(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Action, create_context, tests::compile};
    use indoc::indoc;

    fn lower_source(source: &str, optimize: bool) -> Result<String, Error> {
        let context = create_context();
        let module = compile(&context, source, Action::MlirLlvm, optimize)?;

        assert!(module.as_operation().verify());

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn lower_print() {
        insta::assert_snapshot!(lower_source(include_str!("../examples/jit.toy"), false).unwrap());
    }

    #[test]
    fn lower_multiply_transpose_with_optimization() {
        insta::assert_snapshot!(
            lower_source(include_str!("../examples/codegen.toy"), true).unwrap()
        );
    }

    #[test]
    fn lower_scalar_print() {
        insta::assert_snapshot!(
            lower_source(
                indoc!(
                    "
                    def main() {
                      print(1);
                    }
                    "
                ),
                false
            )
            .unwrap()
        );
    }

    #[test]
    fn lower_two_prints() {
        insta::assert_snapshot!(
            lower_source(
                indoc!(
                    "
                    def main() {
                      var a<2, 2> = [1, 2, 3, 4];
                      print(a);
                      print(transpose(a));
                    }
                    "
                ),
                false
            )
            .unwrap()
        );
    }
}
