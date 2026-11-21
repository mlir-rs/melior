mod function;
mod module;

use self::module::ModuleGenerator;
use crate::{ast, error::Error, location::Location};
use melior::{Context, ir::Module};

/// Generates an MLIR module in the Toy dialect from a Toy AST.
pub fn generate<'c>(context: &'c Context, module: &ast::Module) -> Result<Module<'c>, Error> {
    ModuleGenerator::new(context, module.path()).generate(module)
}

const fn error(location: Location, message: String) -> Error {
    Error::CodeGeneration { location, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{Expression, Function, Literal, Record, Statement, StructLiteral},
        compile::create_context,
        parse::parse,
    };
    use indoc::indoc;

    fn generate_source(source: &str) -> Result<String, Error> {
        let context = create_context();
        let module = generate(&context, &parse(source, "test.toy")?)?;

        Ok(module.as_operation().to_string())
    }

    #[test]
    fn generate_codegen_example() {
        insta::assert_snapshot!(generate_source(include_str!("../examples/codegen.toy")).unwrap());
    }

    #[test]
    fn generate_scalar_example() {
        insta::assert_snapshot!(generate_source(include_str!("../examples/scalar.toy")).unwrap());
    }

    #[test]
    fn generate_struct_example() {
        insta::assert_snapshot!(
            generate_source(include_str!("../examples/struct_codegen.toy")).unwrap()
        );
    }

    #[test]
    fn generate_addition_and_nested_struct() {
        insta::assert_snapshot!(
            generate_source(indoc!(
                "
                struct Inner {
                  var a;
                }

                struct Outer {
                  Inner inner;
                  var b;
                }

                def main() {
                  Outer x = {{[1, 2]}, [3, 4]};
                  print(x.inner.a + x.b);
                  return;
                }
                "
            ))
            .unwrap()
        );
    }

    #[test]
    fn generate_largest_tensor_shape() {
        assert!(generate_source("def main() { var a<1152921504606846975> = 1; }").is_ok());
    }

    #[test]
    fn fail_to_generate_unknown_variable() {
        assert!(matches!(
            generate_source("def main() { print(a); }"),
            Err(Error::CodeGeneration { message, .. }) if message == "unknown variable 'a'"
        ));
    }

    #[test]
    fn fail_to_generate_unknown_function() {
        assert!(matches!(
            generate_source("def main() { foo(1); }"),
            Err(Error::CodeGeneration { message, .. }) if message == "unknown function 'foo'"
        ));
    }

    #[test]
    fn fail_to_generate_transpose_without_arguments() {
        assert!(matches!(
            generate_source("def main() { print(transpose()); }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "function 'transpose' expects exactly one argument"
        ));
    }

    #[test]
    fn fail_to_generate_duplicate_variable() {
        assert!(matches!(
            generate_source("def main() { var a = 1; var a = 2; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "variable 'a' is already declared"
        ));
    }

    #[test]
    fn fail_to_generate_duplicate_struct() {
        assert!(matches!(
            generate_source("struct Foo { var a; } struct Foo { var a; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "struct type 'Foo' is already defined"
        ));
    }

    #[test]
    fn fail_to_generate_subtraction() {
        assert!(matches!(
            generate_source("def main() { var a = 1 - 2; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "invalid binary operator '-'"
        ));
    }

    #[test]
    fn fail_to_generate_invalid_struct_access() {
        assert!(matches!(
            generate_source("def main() { var a = 1; var b = a.c; }"),
            Err(Error::CodeGeneration { message, .. })
                if message == "invalid access into struct expression"
        ));
    }

    #[test]
    fn fail_to_generate_mismatched_struct_initializer() {
        assert!(matches!(
            generate_source("struct Foo { var a; var b; } def main() { Foo x = {1}; }"),
            Err(Error::CodeGeneration { message, .. })
                if message.starts_with("struct type of initializer is different")
        ));
    }

    #[test]
    fn fail_to_generate_unknown_struct() {
        assert!(matches!(
            generate_source("def main() { Foo x = {1}; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "unknown struct type 'Foo'"
        ));
    }

    #[test]
    fn fail_to_generate_dimension_above_maximum() {
        assert!(matches!(
            generate_source("def main() { var a<0, 9223372036854775808> = 1; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_tensor_size_above_maximum() {
        assert!(matches!(
            generate_source("def main() { var a<1152921504606846976> = 1; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_wrapping_tensor_shape() {
        assert!(matches!(
            generate_source("def main() { var a<3, 6148914691236517206> = [1, 2]; }"),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }

    #[test]
    fn fail_to_generate_too_large_literal() {
        let location = Location::new(1, 1);

        assert!(matches!(
            generate(
                &create_context(),
                &ast::Module::new(
                    "test.toy".into(),
                    vec![Record::Function(Function::new(
                        ast::Prototype::new("main".into(), vec![], location),
                        vec![Statement::Expression(Expression::StructLiteral(
                            StructLiteral::new(
                                vec![Expression::Literal(Literal::new(
                                    vec![Expression::Number(ast::Number::new(1.0, location))],
                                    vec![1 << 32, 1 << 32],
                                    location,
                                ))],
                                location,
                            )
                        ))],
                    ))],
                ),
            ),
            Err(Error::CodeGeneration { message, .. }) if message == "tensor shape is too large"
        ));
    }
}
