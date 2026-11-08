mod error;
mod generation;
mod input;
mod operation;
mod r#type;
mod utility;

use self::{
    error::Error,
    generation::{generate_operation, generate_operation_enum},
    utility::{sanitize_documentation, sanitize_snake_case_identifier},
};
pub use input::DialectInput;
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::{
    env,
    path::{Component, Path},
};
use tblgen::{TableGenParser, record::Record, record_keeper::RecordKeeper};

const LLVM_INCLUDE_DIRECTORY: &str = env!("LLVM_INCLUDE_DIRECTORY");

pub fn generate_dialect(input: DialectInput) -> Result<TokenStream, Error> {
    let mut parser = TableGenParser::new();

    parser = parser.add_include_directory(LLVM_INCLUDE_DIRECTORY);

    for path in input.directories() {
        parser = parser.add_include_directory(&resolve_include_directory(path));
    }

    for (env_var, span) in input.directory_env_vars() {
        parser = parser.add_include_directory(&resolve_include_directory(
            &env::var(env_var).map_err(|error| syn::Error::new(span, error.to_string()))?,
        ));
    }

    if input.files().count() > 0 {
        parser = parser.add_source(&input.files().fold(String::new(), |source, path| {
            source + "include \"" + path + "\"\n"
        }))?;
    }

    let keeper = parser.parse().map_err(Error::Parse)?;

    let dialect = generate_dialect_module(
        input.name(),
        keeper
            .all_derived_definitions("Dialect")
            .find(|definition| definition.str_value("name") == Ok(input.name()))
            .ok_or_else(|| syn::Error::new(Span::call_site(), "dialect not found"))?,
        &keeper,
    )
    .map_err(|error| error.add_source_info(keeper.source_info()))?;

    Ok(quote! { #dialect }.into())
}

fn generate_dialect_module(
    name: &str,
    dialect: Record,
    record_keeper: &RecordKeeper,
) -> Result<proc_macro2::TokenStream, Error> {
    let dialect_name = dialect.name()?;

    let all_operations = record_keeper
        .all_derived_definitions("Op")
        .filter(|definition| {
            definition
                .def_value("opDialect")
                .and_then(Record::name)
                .is_ok_and(|name| name == dialect_name)
        })
        .map(Operation::new)
        .collect::<Result<Vec<_>, _>>()?;

    let operations = all_operations
        .iter()
        .map(generate_operation)
        .collect::<Vec<_>>();

    let doc = format!(
        "`{name}` dialect.\n\n{}",
        sanitize_documentation(dialect.str_value("description").unwrap_or(""),)?
    );
    let name = sanitize_snake_case_identifier(name)?;
    let enum_definition = generate_operation_enum(dialect_name, &all_operations);

    Ok(quote! {
        #[doc = #doc]
        pub mod #name {
            use ::melior::ir::operation::OperationLike;
            use ::melior::ir::operation::OperationMutLike;

            #(#operations)*

            #enum_definition
        }
    })
}

fn resolve_include_directory(path: &str) -> String {
    if matches!(
        Path::new(path).components().next(),
        Some(Component::CurDir | Component::ParentDir)
    ) {
        path.into()
    } else {
        Path::new(LLVM_INCLUDE_DIRECTORY).join(path)
    }
    .display()
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_record_keeper() -> RecordKeeper<'static> {
        TableGenParser::new()
            .add_include_directory(LLVM_INCLUDE_DIRECTORY)
            .add_source(&unindent::unindent(
                r#"
                include "mlir/IR/OpBase.td"

                def Foo_Dialect : Dialect {
                    let name = "foo";
                }

                def Bar_Dialect : Dialect {
                    let name = "bar";
                }

                def Foo_BazOp : Op<Foo_Dialect, "baz">;

                def Bar_QuxOp : Op<Bar_Dialect, "qux"> {
                    let arguments = (ins Variadic<I32>:$lhs, Variadic<I32>:$rhs);
                }
                "#,
            ))
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn ignore_invalid_operation_in_other_dialect() {
        let keeper = create_record_keeper();
        let module = generate_dialect_module("foo", keeper.def("Foo_Dialect").unwrap(), &keeper)
            .unwrap()
            .to_string();

        assert!(module.contains("BazOperation"));
        assert!(!module.contains("QuxOperation"));
    }

    #[test]
    fn fail_to_generate_dialect_with_invalid_operation() {
        let keeper = create_record_keeper();

        assert!(matches!(
            generate_dialect_module("bar", keeper.def("Bar_Dialect").unwrap(), &keeper),
            Err(Error::Ods(_))
        ));
    }
}
