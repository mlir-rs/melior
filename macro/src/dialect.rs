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
use std::{env, fmt::Display, path::Path, str};
use tblgen::{record::Record, record_keeper::RecordKeeper, TableGenParser};

pub fn generate_dialect(input: DialectInput) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let mut parser = TableGenParser::new();

    if let Some(source) = input.table_gen() {
        let base = Path::new(env!("LLVM_INCLUDE_DIRECTORY"));

        // Here source looks like " include "somepathto.td" include "somepathto2.td" "
        for (i, p) in source.split_ascii_whitespace().enumerate() {
            if i % 2 == 0 {
                continue; // skip "include"
            }

            let x = &p[1..(p.len() - 1)]; // remove ""
            let path = Path::new(x).parent().unwrap();
            let path = base.join(path);
            parser = parser.add_include_path(&path.to_string_lossy());
        }

        parser = parser.add_source(source).map_err(create_syn_error)?;
    }

    if let Some(file) = input.td_file() {
        parser = parser.add_source_file(file).map_err(create_syn_error)?;
    }

    for path in input
        .include_directories()
        .chain([env!("LLVM_INCLUDE_DIRECTORY")])
    {
        parser = parser.add_include_path(path);
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
            #(#operations)*
        }
    })
}

fn create_syn_error(error: impl Display) -> syn::Error {
    syn::Error::new(Span::call_site(), format!("{}", error))
}
