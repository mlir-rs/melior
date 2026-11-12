//! Intra-procedural shape inference of Toy operations.

use crate::{
    error::Error,
    utility::{block_operations, body_block, is_operation, operation_name},
};
use melior::ir::{
    Module, OperationRef, TypeLike, ValueLike,
    attribute::{AttributeLike, DenseElementsAttribute},
    operation::OperationLike,
    r#type::{DimSize, RankedTensorType, ShapedTypeLike},
};

// TODO Implement this as an operation interface once Toy operations can
// implement interfaces.
/// Infers shapes of Toy operations in every function of a module.
///
/// Operations returning tensors of unknown shapes are collected in a worklist.
/// Then, shapes of the operations are inferred one by one as soon as all of
/// their operands have known shapes until the worklist gets empty.
pub fn infer_shapes(module: &Module) -> Result<(), Error> {
    for function in
        block_operations(module.body()).filter(|operation| is_operation(*operation, "toy.func"))
    {
        let Some(block) = body_block(function) else {
            continue;
        };
        let mut worklist = block_operations(block)
            .filter(|operation| {
                operation
                    .results()
                    .any(|result| !result.r#type().is_ranked_tensor())
            })
            .collect::<Vec<_>>();

        while let Some(index) = worklist.iter().position(|operation| {
            operation
                .operands()
                .all(|operand| operand.r#type().is_ranked_tensor())
        }) {
            infer_shape(worklist.remove(index))?;
        }

        if !worklist.is_empty() {
            return Err(Error::ShapeInference(format!(
                "{} operations couldn't be inferred",
                worklist.len()
            )));
        }
    }

    Ok(())
}

fn infer_shape(operation: OperationRef) -> Result<(), Error> {
    let result = operation.result(0)?;
    let operand_type = |index| Ok::<_, melior::Error>(operation.operand(index)?.r#type());

    result.set_type(match operation_name(operation)?.as_str() {
        "toy.constant" => DenseElementsAttribute::try_from(operation.attribute("value")?)?.r#type(),
        "toy.add" | "toy.mul" => {
            let (lhs, rhs) = (operand_type(0)?, operand_type(1)?);

            if lhs != rhs {
                return Err(Error::ShapeInference(format!(
                    "operand shapes mismatch: {lhs} and {rhs}"
                )));
            }

            lhs
        }
        "toy.cast" => operand_type(0)?,
        "toy.transpose" => {
            let input = RankedTensorType::try_from(operand_type(0)?)?;

            RankedTensorType::new(
                &(0..input.rank())
                    .rev()
                    .map(|index| match input.dim_size(index)? {
                        DimSize::Static(size) => Ok(size),
                        DimSize::Dynamic => Err(Error::ShapeInference(format!(
                            "dynamic dimension in {input}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                input.element(),
                None,
            )
            .into()
        }
        name => {
            return Err(Error::ShapeInference(format!(
                "unable to infer shape of operation '{name}'"
            )));
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        canonicalize::canonicalize, compile::create_context, inliner::inline_functions,
        mlir_gen::generate, parse::parse,
    };
    use indoc::indoc;

    fn infer_source(source: &str) -> Result<String, Error> {
        let context = create_context();
        let mut module = generate(&context, &parse(source, "test.toy")?)?;

        inline_functions(&context, &module)?;
        canonicalize(&context, &mut module)?;
        infer_shapes(&module)?;
        canonicalize(&context, &mut module)?;
        assert!(module.as_operation().verify());

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn infer_multiply_transpose() {
        insta::assert_snapshot!(infer_source(include_str!("../examples/codegen.toy")).unwrap());
    }

    #[test]
    fn infer_struct_example() {
        insta::assert_snapshot!(
            infer_source(include_str!("../examples/struct_codegen.toy")).unwrap()
        );
    }

    #[test]
    fn infer_addition() {
        insta::assert_snapshot!(
            infer_source(indoc!(
                "
                def main() {
                  var a<2, 3> = [1, 2, 3, 4, 5, 6];
                  print(a + a);
                }
                "
            ))
            .unwrap()
        );
    }

    #[test]
    fn fail_to_infer_mismatched_shapes() {
        assert!(matches!(
            infer_source(indoc!(
                "
                def multiply_transpose(a, b) {
                  return transpose(a) * transpose(b);
                }

                def main() {
                  var a<2, 3> = [1, 2, 3, 4, 5, 6];
                  var b<3, 2> = [1, 2, 3, 4, 5, 6];
                  print(multiply_transpose(a, b));
                }
                "
            )),
            Err(Error::ShapeInference(message)) if message.starts_with("operand shapes mismatch")
        ));
    }

    #[test]
    fn fail_to_infer_unknown_operation() {
        let context = create_context();
        let module = melior::ir::Module::parse(
            &context,
            indoc!(
                r#"
                "toy.func"() ({
                  %0 = "toy.constant"() {value = dense<1.0> : tensor<f64>} : () -> tensor<f64>
                  %1 = "toy.unknown"(%0) : (tensor<f64>) -> tensor<*xf64>
                  "toy.return"() : () -> ()
                }) {sym_name = "main", function_type = () -> ()} : () -> ()
                "#
            ),
        )
        .unwrap();

        assert!(matches!(
            infer_shapes(&module),
            Err(Error::ShapeInference(message))
                if message == "unable to infer shape of operation 'toy.unknown'"
        ));
    }

    #[test]
    fn fail_to_infer_non_utf8_operation_name() {
        let context = create_context();
        let module = melior::ir::Module::parse(
            &context,
            indoc!(
                r#"
                "toy.func"() ({
                  %0 = "toy.\FF"() : () -> tensor<*xf64>
                  "toy.return"() : () -> ()
                }) {sym_name = "main", function_type = () -> ()} : () -> ()
                "#
            ),
        )
        .unwrap();

        assert!(matches!(
            infer_shapes(&module),
            Err(Error::Melior(melior::Error::Utf8(_)))
        ));
    }
}
