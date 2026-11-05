mod error;
mod generation;
mod input;
mod operation;
mod r#trait;
mod r#type;
mod utility;

use self::{
    error::Error,
    generation::generate_operation,
    utility::{sanitize_documentation, sanitize_snake_case_identifier},
};
use convert_case::{Case, Casing};
pub use input::DialectInput;
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use std::{
    env,
    path::{Component, Path},
    str,
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
            source + "include \"" + path + "\""
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

fn generate_operation_enum(
    dialect_name: &str,
    operations: &[Operation],
) -> Option<proc_macro2::TokenStream> {
    let enum_identifier = format_ident!("{}Operation", dialect_name.to_case(Case::Pascal));
    let identifiers = operations
        .iter()
        .map(|operation| format_ident!("{}", operation.name()))
        .collect::<Vec<_>>();
    let variants = operations
        .iter()
        .map(|operation| format_ident!("{}", operation.short_name()))
        .collect::<Vec<_>>();
    let full_names = operations.iter().map(Operation::full_operation_name);

    (!operations.is_empty()).then(|| {
        quote! {
            #[derive(Clone, Debug, PartialEq, Eq)]
            pub enum #enum_identifier<'b> {
                #(#variants(#identifiers<'b>)),*
            }

            impl<'b> std::fmt::Display for #enum_identifier<'b> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                   std::fmt::Display::fmt(self.as_operation(), formatter)
                }
            }

            impl<'b> #enum_identifier<'b> {
                pub fn try_new(operation: ::melior::ir::operation::Operation<'b>) -> Result<Self, ::melior::ir::operation::Operation<'b>> {
                    let name = operation.name();
                    let Ok(name_str) = name.as_string_ref().as_str() else {
                        return Err(operation);
                    };
                    match name_str {
                        #(
                            #full_names => Ok(
                                #enum_identifier::#variants(
                                    #identifiers::try_from(operation)
                                        .expect("operation should match type"),
                                ),
                            ),
                        )*
                        _ => Err(operation),
                    }
                }
            }

            impl<'b> #enum_identifier<'b> {
                pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'b> {
                    match self {
                        #(#enum_identifier::#variants(operation) => operation.as_operation(),)*
                    }
                }
            }

            #(
                impl<'b> From<#identifiers<'b>> for #enum_identifier<'b> {
                    fn from(operation: #identifiers<'b>) -> Self {
                        #enum_identifier::#variants(operation)
                    }
                }
            )*
        }
    })
}

fn generate_dialect_module(
    name: &str,
    dialect: Record,
    record_keeper: &RecordKeeper,
) -> Result<proc_macro2::TokenStream, Error> {
    let dialect_name = dialect.name()?;

    let mut all_operations = record_keeper
        .all_derived_definitions("Op")
        .map(Operation::new)
        .collect::<Result<Vec<_>, _>>()?;
    all_operations.retain(|operation| operation.dialect_name() == dialect_name);

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
