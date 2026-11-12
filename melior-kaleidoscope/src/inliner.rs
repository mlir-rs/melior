//! Inlining of calls to Toy functions.

use crate::{
    error::Error,
    toy,
    utility::{block_operations, body_block, is_operation, symbol_name, terminator},
};
use melior::{
    Context, IrRewriter, RewriterBase,
    ir::{
        BlockLike, Module, Operation, OperationRef, ValueLike,
        attribute::{FlatSymbolRefAttribute, StringAttribute},
        operation::OperationLike,
    },
};
use std::collections::{HashMap, HashSet};

// TODO Use the inliner pass once Toy operations can implement its interfaces.
/// Inlines all calls to Toy functions and removes the private functions.
pub fn inline_functions<'c>(context: &'c Context, module: &Module<'c>) -> Result<(), Error> {
    let functions = block_operations(module.body())
        .filter(|operation| is_operation(*operation, "toy.func"))
        .map(|function| Ok((symbol_name(function)?, function)))
        .collect::<Result<Vec<_>, Error>>()?;

    check_recursion(&functions)?;

    let rewriter = IrRewriter::new(context);
    let rewriter = rewriter.as_rewriter_base();
    let function_map = functions.iter().copied().collect::<HashMap<_, _>>();

    while let Some(call) = functions.iter().find_map(|(_, function)| {
        block_operations(body_block(*function)?)
            .find(|operation| is_operation(*operation, "toy.generic_call"))
    }) {
        inline_call(context, &rewriter, &function_map, call)?;
    }

    // Functions other than the main one are private and no longer called.
    for (_, function) in functions {
        if function.attribute("sym_visibility").ok()
            == Some(StringAttribute::new(context, "private").into())
        {
            rewriter.erase_op(function);
        }
    }

    Ok(())
}

// Toy has no control flow, so a recursive function never terminates and
// cannot be inlined.
fn check_recursion(functions: &[(&str, OperationRef)]) -> Result<(), Error> {
    let callees = functions
        .iter()
        .map(|(name, function)| {
            Ok((
                *name,
                body_block(*function)
                    .into_iter()
                    .flat_map(block_operations)
                    .filter(|operation| is_operation(*operation, "toy.generic_call"))
                    .map(callee_name)
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        })
        .collect::<Result<HashMap<_, _>, Error>>()?;

    for (name, _) in functions {
        if reaches(&callees, name, name, &mut HashSet::new()) {
            return Err(Error::RecursiveFunction((*name).into()));
        }
    }

    Ok(())
}

fn reaches<'a>(
    callees: &HashMap<&'a str, Vec<&'a str>>,
    from: &str,
    to: &str,
    visited: &mut HashSet<&'a str>,
) -> bool {
    callees.get(from).into_iter().flatten().any(|callee| {
        *callee == to || (visited.insert(callee) && reaches(callees, callee, to, visited))
    })
}

fn callee_name<'c>(call: OperationRef<'c, '_>) -> Result<&'c str, Error> {
    Ok(FlatSymbolRefAttribute::try_from(call.attribute("callee")?)?.value()?)
}

fn inline_call<'c>(
    context: &'c Context,
    rewriter: &RewriterBase<'c, '_>,
    functions: &HashMap<&str, OperationRef<'c, '_>>,
    call: OperationRef<'c, '_>,
) -> Result<(), Error> {
    let callee_name = callee_name(call)?;
    let callee = functions
        .get(callee_name)
        .ok_or_else(|| Error::FunctionNotFound(callee_name.into()))?;
    // The callee is cloned so that it can be inlined at several call sites.
    let callee: Operation = (**callee).clone();
    let block = body_block(&callee).ok_or_else(|| Error::InvalidFunction(callee_name.into()))?;

    if block.argument_count() != call.operand_count() {
        return Err(Error::ArgumentCount {
            function: callee_name.into(),
            expected: block.argument_count(),
            actual: call.operand_count(),
        });
    }

    rewriter.set_insertion_point_before(call);

    let arguments = (0..call.operand_count())
        .map(|index| {
            let operand = call.operand(index)?;
            let argument = block.argument(index)?;

            // A call operand is cast into a callee argument type when they
            // differ, for example, from a ranked tensor into an unranked one.
            Ok(if operand.r#type() == argument.r#type() {
                operand
            } else {
                rewriter
                    .insert(toy::cast(context, argument.r#type(), operand, call.location()).into())
                    .result(0)?
                    .into()
            })
        })
        .collect::<Result<Vec<_>, melior::Error>>()?;
    let terminator = terminator(block)
        .filter(|terminator| is_operation(*terminator, "toy.return"))
        .ok_or_else(|| Error::InvalidFunction(callee_name.into()))?;

    rewriter.inline_block_before(block, call, &arguments);

    // The inlined return operation is replaced by its operands at the call
    // site.
    let results = terminator.operands().collect::<Vec<_>>();

    if results.len() != call.result_count() {
        return Err(Error::ArgumentCount {
            function: callee_name.into(),
            expected: call.result_count(),
            actual: results.len(),
        });
    }

    rewriter.replace_op_with_values(call, &results);
    rewriter.erase_op(terminator);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile::create_context, mlir_gen::generate, parse::parse};
    use indoc::indoc;

    fn inline_source(source: &str) -> Result<String, Error> {
        let context = create_context();
        let module = generate(&context, &parse(source, "test.toy")?)?;

        inline_functions(&context, &module)?;
        assert!(module.as_operation().verify());

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn inline_generic_calls() {
        insta::assert_snapshot!(inline_source(include_str!("../examples/codegen.toy")).unwrap());
    }

    #[test]
    fn inline_struct_argument() {
        insta::assert_snapshot!(
            inline_source(include_str!("../examples/struct_codegen.toy")).unwrap()
        );
    }

    #[test]
    fn inline_nested_calls() {
        insta::assert_snapshot!(
            inline_source(indoc!(
                "
                def identity(x) {
                  return x;
                }

                def double_transpose(x) {
                  return transpose(identity(transpose(x)));
                }

                def main() {
                  print(double_transpose([[1, 2], [3, 4]]));
                }
                "
            ))
            .unwrap()
        );
    }

    #[test]
    fn remove_unused_private_function() {
        insta::assert_snapshot!(
            inline_source(indoc!(
                "
                def unused(x) {
                  return x;
                }

                def main() {
                  print([1]);
                }
                "
            ))
            .unwrap()
        );
    }

    // Recursive functions cannot be written in Toy because functions must be
    // defined before their uses. They can still appear in MLIR inputs.
    fn inline_module(source: &str) -> Result<(), Error> {
        let context = create_context();
        let module = Module::parse(&context, source).unwrap();

        inline_functions(&context, &module)
    }

    #[test]
    fn fail_to_inline_recursive_function() {
        assert!(matches!(
            inline_module(indoc!(
                r#"
                "toy.func"() ({
                ^bb0(%arg0: tensor<*xf64>):
                  %0 = "toy.generic_call"(%arg0) {callee = @forever} : (tensor<*xf64>) -> tensor<*xf64>
                  "toy.return"(%0) : (tensor<*xf64>) -> ()
                }) {function_type = (tensor<*xf64>) -> tensor<*xf64>, sym_name = "forever", sym_visibility = "private"} : () -> ()
                "#
            )),
            Err(Error::RecursiveFunction(name)) if name == "forever"
        ));
    }

    #[test]
    fn fail_to_inline_mutually_recursive_functions() {
        assert!(matches!(
            inline_module(indoc!(
                r#"
                "toy.func"() ({
                ^bb0(%arg0: tensor<*xf64>):
                  %0 = "toy.generic_call"(%arg0) {callee = @pong} : (tensor<*xf64>) -> tensor<*xf64>
                  "toy.return"(%0) : (tensor<*xf64>) -> ()
                }) {function_type = (tensor<*xf64>) -> tensor<*xf64>, sym_name = "ping", sym_visibility = "private"} : () -> ()
                "toy.func"() ({
                ^bb0(%arg0: tensor<*xf64>):
                  %0 = "toy.generic_call"(%arg0) {callee = @ping} : (tensor<*xf64>) -> tensor<*xf64>
                  "toy.return"(%0) : (tensor<*xf64>) -> ()
                }) {function_type = (tensor<*xf64>) -> tensor<*xf64>, sym_name = "pong", sym_visibility = "private"} : () -> ()
                "#
            )),
            Err(Error::RecursiveFunction(name)) if name == "ping"
        ));
    }

    #[test]
    fn fail_to_inline_call_with_wrong_argument_count() {
        assert!(matches!(
            inline_module(indoc!(
                r#"
                "toy.func"() ({
                ^bb0(%arg0: tensor<*xf64>):
                  "toy.return"(%arg0) : (tensor<*xf64>) -> ()
                }) {function_type = (tensor<*xf64>) -> tensor<*xf64>, sym_name = "identity", sym_visibility = "private"} : () -> ()
                "toy.func"() ({
                  %0 = "toy.generic_call"() {callee = @identity} : () -> tensor<*xf64>
                  "toy.return"() : () -> ()
                }) {function_type = () -> (), sym_name = "main"} : () -> ()
                "#
            )),
            Err(Error::ArgumentCount { function, expected: 1, actual: 0 }) if function == "identity"
        ));
    }

    #[test]
    fn fail_to_inline_function_without_name() {
        assert!(matches!(
            inline_module(indoc!(
                r#"
                "toy.func"() ({
                  "toy.return"() : () -> ()
                }) {function_type = () -> ()} : () -> ()
                "#
            )),
            Err(Error::Melior(melior::Error::AttributeNotFound(name))) if name == "sym_name"
        ));
    }

    #[test]
    fn fail_to_inline_function_with_non_string_name() {
        assert!(matches!(
            inline_module(indoc!(
                r#"
                "toy.func"() ({
                  "toy.return"() : () -> ()
                }) {function_type = () -> (), sym_name = 42 : i64} : () -> ()
                "#
            )),
            Err(Error::Melior(melior::Error::AttributeExpected(..)))
        ));
    }
}
