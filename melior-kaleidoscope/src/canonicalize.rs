//! Canonicalization of Toy operations.

use crate::{
    error::Error,
    toy,
    utility::{block_operations, body_block, defining_operation, is_operation, operation_name},
};
use melior::{
    Context, GreedyRewriteDriverConfig, IrRewriter, PatternRewriter, RewritePatternSet,
    RewriterBase, apply_patterns_and_fold_greedily, create_op_rewrite_pattern,
    ir::{
        BlockRef, Module, OperationRef, TypeLike, Value, ValueLike,
        attribute::{ArrayAttribute, AttributeLike, DenseElementsAttribute, IntegerAttribute},
        operation::OperationLike,
    },
};
use std::collections::HashMap;

const PURE_OPERATIONS: &[&str] = &[
    "toy.add",
    "toy.cast",
    "toy.constant",
    "toy.mul",
    "toy.reshape",
    "toy.struct_access",
    "toy.struct_constant",
    "toy.transpose",
];

/// Canonicalizes Toy operations in a module.
pub fn canonicalize<'c>(context: &'c Context, module: &mut Module<'c>) -> Result<(), Error> {
    let patterns = RewritePatternSet::new(context);

    patterns.add(create_op_rewrite_pattern(
        "toy.transpose",
        1,
        context,
        simplify_redundant_transpose,
        &[],
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.reshape",
        1,
        context,
        simplify_reshape,
        &["toy.constant", "toy.reshape"],
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.cast",
        1,
        context,
        fold_cast,
        &[],
    ));
    patterns.add(create_op_rewrite_pattern(
        "toy.struct_access",
        1,
        context,
        fold_struct_access,
        &["toy.constant", "toy.struct_constant"],
    ));

    apply_patterns_and_fold_greedily(
        module,
        &patterns.freeze(),
        &GreedyRewriteDriverConfig::new(),
    )?;

    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();

    for function in block_operations(module.body()) {
        if let Some(block) = body_block(function) {
            eliminate_common_subexpressions(&rewriter, block);
            remove_dead_operations(&rewriter, block);
        }
    }

    Ok(())
}

// Toy operations are not registered in MLIR, so the common sub-expression
// elimination pass cannot know that they are free of side effects. This
// replaces uses of pure Toy operations equivalent to earlier ones in a block
// by the results of the earlier ones instead.
fn eliminate_common_subexpressions<'c>(rewriter: &RewriterBase<'c, '_>, block: BlockRef<'c, '_>) {
    let mut operations = HashMap::<_, Vec<Value<'c, '_>>>::new();

    for operation in block_operations(block) {
        if !is_pure(operation) {
            continue;
        }

        // Attributes and types are unique within a context.
        let key = (
            operation_name(operation),
            operation.operands().map(value_id).collect::<Vec<_>>(),
            operation
                .attributes()
                .map(|(name, attribute)| {
                    (
                        name.as_string_ref().as_str().unwrap_or_default().to_owned(),
                        attribute.to_raw().ptr as usize,
                    )
                })
                .collect::<Vec<_>>(),
            operation
                .results()
                .map(|result| result.r#type().to_raw().ptr as usize)
                .collect::<Vec<_>>(),
        );
        let results = operation.results().map(Into::into).collect::<Vec<_>>();

        if let Some(existing) = operations.get(&key) {
            for (result, existing) in results.iter().zip(existing) {
                rewriter.replace_all_uses_with(*result, *existing);
            }
        } else {
            operations.insert(key, results);
        }
    }
}

fn is_pure<'c: 'a, 'a>(operation: OperationRef<'c, 'a>) -> bool {
    PURE_OPERATIONS
        .iter()
        .any(|name| is_operation(operation, name))
}

// transpose(transpose(x)) -> x
fn simplify_redundant_transpose<'c>(
    operation: OperationRef<'c, '_>,
    rewriter: &PatternRewriter<'c, '_>,
) -> bool {
    let Some(input) = operation
        .operand(0)
        .ok()
        .and_then(defining_operation)
        .filter(|operation| is_operation(*operation, "toy.transpose"))
        .and_then(|operation| operation.operand(0).ok())
    else {
        return false;
    };

    rewriter
        .as_rewriter_base()
        .replace_op_with_values(operation, &[input]);

    true
}

fn simplify_reshape<'c>(
    operation: OperationRef<'c, '_>,
    rewriter: &PatternRewriter<'c, '_>,
) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let (Ok(input), Ok(output)) = (operation.operand(0), operation.result(0)) else {
        return false;
    };
    let output_type = output.r#type();

    // A reshape into the same type is redundant.
    if input.r#type() == output_type {
        rewriter.replace_op_with_values(operation, &[input]);
        return true;
    }

    let Some(producer) = defining_operation(input) else {
        return false;
    };

    let replacement = if is_operation(producer, "toy.reshape") {
        // reshape(reshape(x)) -> reshape(x)
        let Ok(input) = producer.operand(0) else {
            return false;
        };

        toy::reshape(context, output_type, input, operation.location()).into()
    } else if is_operation(producer, "toy.constant") {
        // reshape(constant) -> constant
        //
        // A splat constant, such as a single number, is broadcast to the
        // output shape.
        let Ok(value) = producer
            .attribute("value")
            .and_then(DenseElementsAttribute::try_from)
            .and_then(|value| {
                if value.is_splat() {
                    DenseElementsAttribute::splat(output_type, value.splat_value()?)
                } else {
                    value.reshape(output_type)
                }
            })
        else {
            return false;
        };

        toy::constant(context, output_type, value, operation.location()).into()
    } else {
        return false;
    };
    let replacement = rewriter.insert(replacement);

    rewriter.replace_op_with_operation(operation, replacement);

    true
}

// A cast into the same type is folded away.
fn fold_cast<'c>(operation: OperationRef<'c, '_>, rewriter: &PatternRewriter<'c, '_>) -> bool {
    let (Ok(input), Ok(output)) = (operation.operand(0), operation.result(0)) else {
        return false;
    };

    if input.r#type() != output.r#type() {
        return false;
    }

    rewriter
        .as_rewriter_base()
        .replace_op_with_values(operation, &[input]);

    true
}

// An access into a constant struct is folded into the accessed constant.
fn fold_struct_access<'c>(
    operation: OperationRef<'c, '_>,
    rewriter: &PatternRewriter<'c, '_>,
) -> bool {
    let rewriter = rewriter.as_rewriter_base();
    let context = rewriter.context();
    // SAFETY: The context reference lives until the end of this function.
    let context = unsafe { context.to_ref() };
    let Some(producer) = operation
        .operand(0)
        .ok()
        .and_then(defining_operation)
        .filter(|operation| is_operation(*operation, "toy.struct_constant"))
    else {
        return false;
    };
    let (Ok(value), Ok(index), Ok(output)) = (
        producer
            .attribute("value")
            .and_then(ArrayAttribute::try_from),
        operation
            .attribute("index")
            .and_then(IntegerAttribute::try_from),
        operation.result(0),
    ) else {
        return false;
    };
    let Ok(element) = value.element(index.value() as usize) else {
        return false;
    };
    let output_type = output.r#type();
    let location = operation.location();

    let constant = if let Ok(element) = DenseElementsAttribute::try_from(element) {
        toy::constant(context, output_type, element, location).into()
    } else if let Ok(element) = ArrayAttribute::try_from(element) {
        toy::struct_constant(context, output_type, element, location).into()
    } else {
        return false;
    };
    let constant = rewriter.insert(constant);

    rewriter.replace_op_with_operation(operation, constant);

    true
}

// Toy operations are not registered in MLIR, so the greedy rewrite driver
// cannot know that the unused ones are free of side effects. This removes
// unused pure Toy operations instead.
fn remove_dead_operations<'c>(rewriter: &RewriterBase<'c, '_>, block: BlockRef<'c, '_>) {
    let operations = block_operations(block).collect::<Vec<_>>();
    let mut use_counts = HashMap::<usize, usize>::new();

    for operation in &operations {
        for operand in operation.operands() {
            *use_counts.entry(value_id(operand)).or_default() += 1;
        }
    }

    // Uses of a value always follow its definition in a block.
    for operation in operations.into_iter().rev() {
        if !is_pure(operation)
            || operation
                .results()
                .any(|result| use_counts.contains_key(&value_id(result.into())))
        {
            continue;
        }

        for operand in operation.operands() {
            if let Some(count) = use_counts.get_mut(&value_id(operand)) {
                *count -= 1;

                if *count == 0 {
                    use_counts.remove(&value_id(operand));
                }
            }
        }

        rewriter.erase_op(operation);
    }
}

fn value_id(value: Value) -> usize {
    value.to_raw().ptr as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile::create_context, mlir_gen::generate, parse::parse};
    use indoc::indoc;

    fn canonicalize_source(source: &str) -> String {
        let context = create_context();
        let mut module = generate(&context, &parse(source, "test.toy").unwrap()).unwrap();

        canonicalize(&context, &mut module).unwrap();
        assert!(module.as_operation().verify());

        module.as_operation().to_string()
    }

    #[test]
    fn simplify_transpose_of_transpose() {
        insta::assert_snapshot!(canonicalize_source(indoc!(
            "
            def transpose_transpose(x) {
              return transpose(transpose(x));
            }
            "
        )));
    }

    #[test]
    fn simplify_reshapes_of_constant() {
        insta::assert_snapshot!(canonicalize_source(include_str!(
            "../examples/trivial_reshape.toy"
        )));
    }

    #[test]
    fn simplify_reshape_of_reshape() {
        insta::assert_snapshot!(canonicalize_source(indoc!(
            "
            def reshape(x) {
              var a<2, 3> = x;
              var b<3, 2> = a;
              return b;
            }
            "
        )));
    }

    #[test]
    fn fold_access_into_struct_constant() {
        insta::assert_snapshot!(canonicalize_source(indoc!(
            "
            struct Struct {
              var a;
              var b;
            }

            def main() {
              Struct value = {[[1, 2, 3], [4, 5, 6]], [1, 2]};
              print(value.a);
              print(value.b);
            }
            "
        )));
    }

    #[test]
    fn eliminate_common_constants_of_same_type() {
        insta::assert_snapshot!(canonicalize_source(indoc!(
            "
            def main() {
              var a<2, 3> = [1, 2, 3, 4, 5, 6];
              var b<2, 3> = [[1, 2, 3], [4, 5, 6]];
              var c<3, 2> = [1, 2, 3, 4, 5, 6];
              print(a);
              print(b);
              print(c);
            }
            "
        )));
    }

    #[test]
    fn eliminate_common_transposes() {
        insta::assert_snapshot!(canonicalize_source(indoc!(
            "
            def main() {
              var a<2, 3> = [1, 2, 3, 4, 5, 6];
              print(transpose(a) * transpose(a));
              print(transpose(a) + transpose(transpose(a)));
            }
            "
        )));
    }

    #[test]
    fn keep_used_operations() {
        insta::assert_snapshot!(canonicalize_source(include_str!("../examples/codegen.toy")));
    }
}
