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
pub use input::DialectInput;
use operation::Operation;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::{
    env,
    fmt::Display,
    path::{Component, Path},
    str,
};
use tblgen::{TableGenParser, record::Record, record_keeper::RecordKeeper};

const LLVM_INCLUDE_DIRECTORY: &str = env!("LLVM_INCLUDE_DIRECTORY");

pub fn generate_dialect(input: DialectInput) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut parser = TableGenParser::new();

    for path in input.include_directories().chain([LLVM_INCLUDE_DIRECTORY]) {
        parser = parser.add_include_directory(path);
    }

    for path in input.directories() {
        let path = if matches!(
            Path::new(path).components().next(),
            Some(Component::CurDir | Component::ParentDir)
        ) {
            path.into()
        } else {
            Path::new(LLVM_INCLUDE_DIRECTORY).join(path)
        };

        parser = parser.add_include_directory(&path.display().to_string());
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
            .ok_or_else(|| create_syn_error("dialect not found"))?,
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
    let operations = record_keeper
        .all_derived_definitions("Op")
        .map(Operation::new)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .filter(|operation| operation.dialect_name() == dialect_name)
        .map(generate_operation)
        .collect::<Vec<_>>();

    let doc = format!(
        "`{name}` dialect.\n\n{}",
        sanitize_documentation(dialect.str_value("description").unwrap_or(""),)?
    );
    let name = sanitize_snake_case_identifier(name)?;

    Ok(quote! {
        #[doc = #doc]
        pub mod #name {
            use melior::ir::operation::OperationLike;
            use melior::ir::operation::OperationMutLike;

            #(#operations)*
        }
    })
}

fn create_syn_error(error: impl Display) -> syn::Error {
    syn::Error::new(Span::call_site(), format!("{error}"))
}
