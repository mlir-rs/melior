//! Partial lowering of Toy operations to the affine, arith, func, and memref
//! dialects.
//!
//! Print operations are kept in the Toy dialect while their tensor operands
//! are replaced by memory references.

use crate::{
    error::Error,
    utility::{check_lowering, is_operation, terminator},
};
use melior::{
    Context, GreedyRewriteDriverConfig, PatternRewriter, RewritePatternSet, RewriterBase,
    apply_patterns_and_fold_greedily, create_op_rewrite_pattern,
    dialect::ods::{affine, arith, func, memref},
    ir::{
        AffineExpr, AffineMap, Attribute, Block, BlockLike, Location, Module, Operation,
        OperationRef, Region, RegionLike, Type, TypeLike, Value, ValueLike,
        attribute::{
            DenseElementsAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationLike,
        r#type::{DimSize, FunctionType, MemRefType, RankedTensorType, ShapedTypeLike},
    },
};

/// Lowers Toy operations except print operations in a module to affine loops.
pub fn lower_to_affine<'c>(context: &'c Context, module: &mut Module<'c>) -> Result<(), Error> {
    let patterns = RewritePatternSet::new(context);
    let memory_operations = ["memref.alloc", "memref.dealloc"];
    let loop_operations = [
        "memref.alloc",
        "memref.dealloc",
        "affine.for",
        "affine.load",
        "affine.store",
        "affine.yield",
    ];

    patterns.add(create_op_rewrite_pattern(
        "toy.constant",
        1,
        context,
        lower_constant,
        &[&memory_operations[..], &["arith.constant", "affine.store"]].concat(),
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.transpose",
        1,
        context,
        lower_transpose,
        &loop_operations,
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.add",
        1,
        context,
        |operation, rewriter| {
            lower_binary(operation, rewriter, |context, lhs, rhs, location| {
                arith::addf(context, lhs, rhs, location).into()
            })
        },
        &[&loop_operations[..], &["arith.addf"]].concat(),
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.mul",
        1,
        context,
        |operation, rewriter| {
            lower_binary(operation, rewriter, |context, lhs, rhs, location| {
                arith::mulf(context, lhs, rhs, location).into()
            })
        },
        &[&loop_operations[..], &["arith.mulf"]].concat(),
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.func",
        1,
        context,
        lower_function,
        &["func.func"],
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.return",
        1,
        context,
        lower_return,
        &["func.return"],
    ));

    let config = GreedyRewriteDriverConfig::new();
    // Producers are visited before consumers so that operands are already
    // lowered to memory references in most cases.
    config.set_use_top_down_traversal(true);

    apply_patterns_and_fold_greedily(module, &patterns.freeze(), &config)?;

    // Only print operations with memory reference operands are left in the
    // Toy dialect.
    check_lowering(module, |operation| {
        is_operation(operation, "toy.print")
            && operation
                .operands()
                .all(|operand| !operand.r#type().is_tensor())
    })
}

fn lower_constant<'c>(operation: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let Ok(elements) = operation
        .attribute("value")
        .and_then(DenseElementsAttribute::try_from)
        .and_then(|value| {
            (0..value.len())
                .map(|index| value.f64_element(index))
                .collect::<Result<Vec<_>, _>>()
        })
    else {
        return false;
    };
    let Some((alloc, shape)) = allocate_result(context, &rewriter, operation) else {
        return false;
    };
    let location = operation.location();
    let element_type = Type::float64(context);

    // Each element is stored at its constant indices.
    for (index, element) in elements.into_iter().enumerate() {
        let Ok(constant) = rewriter
            .insert(
                arith::constant(
                    context,
                    element_type,
                    FloatAttribute::new(context, element_type, element).into(),
                    location,
                )
                .into(),
            )
            .result(0)
        else {
            return false;
        };

        rewriter.insert(
            affine::store(
                context,
                constant.into(),
                alloc,
                &[],
                affine_map_attribute(
                    context,
                    AffineMap::new(
                        context,
                        0,
                        0,
                        &unravel_index(index, &shape)
                            .into_iter()
                            .map(|index| AffineExpr::constant(context, index as i64))
                            .collect::<Vec<_>>(),
                    ),
                ),
                location,
            )
            .into(),
        );
    }

    rewriter.replace_op_with_values(operation, &[alloc]);

    true
}

// Converts a flat index into multi-dimensional indices in the row-major order.
fn unravel_index(mut index: usize, shape: &[u64]) -> Vec<u64> {
    let mut indices = vec![0; shape.len()];

    for (dimension, &size) in shape.iter().enumerate().rev() {
        indices[dimension] = index as u64 % size;
        index /= size as usize;
    }

    indices
}

fn lower_transpose<'c>(
    operation: OperationRef<'c, '_>,
    rewriter: &PatternRewriter<'c, '_>,
) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let Some(input) = operation
        .operand(0)
        .ok()
        .filter(|input| input.r#type().is_mem_ref())
    else {
        return false;
    };

    lower_to_loops(
        context,
        &rewriter,
        operation,
        |rewriter, induction_variables| {
            // Elements are loaded from the reversed indices.
            load_element(
                context,
                rewriter,
                input,
                &induction_variables
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>(),
                operation.location(),
            )
        },
    )
}

fn lower_binary<'c>(
    operation: OperationRef<'c, '_>,
    rewriter: &PatternRewriter<'c, '_>,
    build: impl Fn(&'c Context, Value<'c, '_>, Value<'c, '_>, Location<'c>) -> Operation<'c>,
) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let (Ok(lhs), Ok(rhs)) = (operation.operand(0), operation.operand(1)) else {
        return false;
    };

    if !lhs.r#type().is_mem_ref() || !rhs.r#type().is_mem_ref() {
        return false;
    }

    lower_to_loops(
        context,
        &rewriter,
        operation,
        |rewriter, induction_variables| {
            let location = operation.location();
            let lhs = load_element(context, rewriter, lhs, induction_variables, location)?;
            let rhs = load_element(context, rewriter, rhs, induction_variables, location)?;

            rewriter
                .insert(build(context, lhs, rhs, location))
                .result(0)
                .ok()
                .map(Into::into)
        },
    )
}

// Lowers an operation to a nest of affine loops storing an element computed by
// the given callback at every index of a result tensor.
fn lower_to_loops<'c, 'r>(
    context: &'c Context,
    rewriter: &'r RewriterBase<'c, '_>,
    operation: OperationRef<'c, '_>,
    build_element: impl FnOnce(&'r RewriterBase<'c, '_>, &[Value<'c, 'r>]) -> Option<Value<'c, 'r>>,
) -> bool {
    let Some((alloc, shape)) = allocate_result(context, rewriter, operation) else {
        return false;
    };
    let location = operation.location();
    let index_type = Type::index(context);
    let Some(induction_variables) = shape
        .iter()
        .map(|&size| {
            let region = Region::new();
            let block = Block::new(&[index_type], &[location]).ok()?;

            block.append_operation(affine::r#yield(context, &[], location).into());
            region.append_block(block);

            let body = rewriter
                .insert(
                    affine::r#for(
                        context,
                        &[],
                        &[],
                        &[],
                        &[],
                        region,
                        affine_map_attribute(context, AffineMap::constant(context, 0)),
                        affine_map_attribute(context, AffineMap::constant(context, size as i64)),
                        IntegerAttribute::new(index_type, 1),
                        location,
                    )
                    .into(),
                )
                .region(0)
                .ok()?
                .first_block()?;

            rewriter.set_insertion_point_before(body.terminator()?);

            Some(body.argument(0).ok()?.into())
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(element) = build_element(rewriter, &induction_variables) else {
        return false;
    };

    rewriter.insert(
        affine::store(
            context,
            element,
            alloc,
            &induction_variables,
            affine_map_attribute(context, AffineMap::multi_dim_identity(context, shape.len())),
            location,
        )
        .into(),
    );
    rewriter.replace_op_with_values(operation, &[alloc]);

    true
}

fn load_element<'c, 'r>(
    context: &'c Context,
    rewriter: &'r RewriterBase<'c, '_>,
    memref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    location: Location<'c>,
) -> Option<Value<'c, 'r>> {
    rewriter
        .insert(
            affine::load(
                context,
                MemRefType::try_from(memref.r#type()).ok()?.element(),
                memref,
                indices,
                affine_map_attribute(
                    context,
                    AffineMap::multi_dim_identity(context, indices.len()),
                ),
                location,
            )
            .into(),
        )
        .result(0)
        .ok()
        .map(Into::into)
}

// Allocates a memory reference for a result tensor of an operation at the
// beginning of its block and deallocates it before the terminator. This is fine
// as Toy functions have no control flow.
fn allocate_result<'c, 'r>(
    context: &'c Context,
    rewriter: &'r RewriterBase<'c, '_>,
    operation: OperationRef<'c, '_>,
) -> Option<(Value<'c, 'r>, Vec<u64>)> {
    let result_type = RankedTensorType::try_from(operation.result(0).ok()?.r#type()).ok()?;
    let shape = (0..result_type.rank())
        .map(|index| match result_type.dim_size(index) {
            Ok(DimSize::Static(size)) => Some(size),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let block = operation.block()?;
    let location = operation.location();

    rewriter.set_insertion_point_to_start(block);

    let alloc = rewriter
        .insert(
            memref::alloc(
                context,
                MemRefType::new(
                    result_type.element(),
                    &shape.iter().map(|&size| size as i64).collect::<Vec<_>>(),
                    None,
                    None,
                )
                .into(),
                &[],
                &[],
                location,
            )
            .into(),
        )
        .result(0)
        .ok()?
        .into();

    rewriter.set_insertion_point_before(terminator(block)?);
    rewriter.insert(memref::dealloc(context, alloc, location).into());
    rewriter.set_insertion_point_before(operation);

    Some((alloc, shape))
}

fn affine_map_attribute<'c>(context: &'c Context, map: AffineMap<'c>) -> Attribute<'c> {
    Attribute::parse(context, &format!("affine_map<{map}>")).expect("valid affine map attribute")
}

// Only the main function is lowered as the other ones are expected to be
// inlined.
fn lower_function<'c>(operation: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let (Ok(name), Ok(function_type), Ok(body)) = (
        operation
            .attribute("sym_name")
            .and_then(StringAttribute::try_from),
        operation
            .attribute("function_type")
            .and_then(TypeAttribute::try_from)
            .and_then(|attribute| FunctionType::try_from(attribute.value())),
        Block::new(&[], &[]),
    ) else {
        return false;
    };
    let Some(old_block) = operation
        .region(0)
        .ok()
        .and_then(|region| region.first_block())
    else {
        return false;
    };

    if Attribute::from(name) != StringAttribute::new(context, "main").into()
        || function_type.input_count() != 0
        || function_type.result_count() != 0
    {
        return false;
    }

    let region = Region::new();
    region.append_block(body);

    let Some(new_block) = rewriter
        .insert(
            func::func(
                context,
                region,
                name,
                TypeAttribute::new(function_type.into()),
                operation.location(),
            )
            .into(),
        )
        .region(0)
        .ok()
        .and_then(|region| region.first_block())
    else {
        return false;
    };

    rewriter.merge_blocks(old_block, new_block, &[]);
    rewriter.erase_op(operation);

    true
}

// All calls are expected to be inlined, so a return operation never has
// operands.
fn lower_return<'c>(operation: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> bool {
    if operation.operand_count() != 0 {
        return false;
    }

    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let r#return = rewriter.insert(func::r#return(context, &[], operation.location()).into());

    rewriter.replace_op_with_operation(operation, r#return);

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Action, create_context, tests::compile};
    use indoc::indoc;

    fn lower_source(source: &str, optimize: bool) -> Result<String, Error> {
        let context = create_context();
        let module = compile(&context, source, Action::MlirAffine, optimize)?;

        assert!(module.as_operation().verify());

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn unravel_indices() {
        assert_eq!(unravel_index(0, &[]), vec![]);
        assert_eq!(unravel_index(0, &[3]), vec![0]);
        assert_eq!(unravel_index(2, &[3]), vec![2]);
        assert_eq!(unravel_index(4, &[2, 3]), vec![1, 1]);
        assert_eq!(unravel_index(5, &[2, 3]), vec![1, 2]);
        assert_eq!(unravel_index(7, &[2, 2, 2]), vec![1, 1, 1]);
    }

    #[test]
    fn lower_multiply_transpose() {
        insta::assert_snapshot!(
            lower_source(include_str!("../examples/codegen.toy"), false).unwrap()
        );
    }

    #[test]
    fn lower_multiply_transpose_with_optimization() {
        insta::assert_snapshot!(
            lower_source(include_str!("../examples/codegen.toy"), true).unwrap()
        );
    }

    #[test]
    fn lower_scalar() {
        insta::assert_snapshot!(
            lower_source(include_str!("../examples/scalar.toy"), false).unwrap()
        );
    }

    #[test]
    fn lower_addition_of_rank_three_tensors() {
        insta::assert_snapshot!(
            lower_source(
                indoc!(
                    "
                    def main() {
                      var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
                      print(a + a);
                    }
                    "
                ),
                false
            )
            .unwrap()
        );
    }

    #[test]
    fn fail_to_lower_reshape() {
        assert!(matches!(
            lower_source(
                indoc!(
                    "
                    def main() {
                      var a<2, 3> = [1, 2, 3, 4, 5, 6];
                      var b<6> = transpose(a) + transpose(a);
                      print(b);
                    }
                    "
                ),
                false
            ),
            Err(Error::Lowering(message)) if message.starts_with("'toy.reshape'")
        ));
    }

    #[test]
    fn fail_to_lower_main_with_result() {
        assert!(matches!(
            lower_source("def main() { return [1]; }", false),
            Err(Error::Lowering(message)) if message.starts_with("'toy.func'")
        ));
    }

    #[test]
    fn fail_to_lower_non_utf8_operation_name() {
        let context = create_context();
        let mut module = melior::ir::Module::parse(
            &context,
            indoc!(
                r#"
                "toy.func"() ({
                  "toy.\FF"() : () -> ()
                  "toy.return"() : () -> ()
                }) {sym_name = "main", function_type = () -> ()} : () -> ()
                "#
            ),
        )
        .unwrap();

        assert!(matches!(
            lower_to_affine(&context, &mut module),
            Err(Error::Melior(melior::Error::Utf8(_)))
        ));
    }
}
