mod attribute;
mod dialect;
mod operation;
mod parse;
mod pass;
mod r#type;
mod utility;

use dialect::DialectInput;
use parse::{DialectOperationSet, IdentifierList, PassSet};
use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Generates a dialect module from a TableGen file.
///
/// # Examples
///
/// ```rust
/// melior::dialect! {
///     name: "func",
///     files: ["IR/FuncOps.td", "TransformOps/FuncTransformOps.td", "Transforms/Passes.td"],
///     include_directories: ["mlir/Dialect/Func"],
///     include_directory_env_vars: ["ENV_VAR_FOR_MLIR_DIALECT_DIR"],
/// }
/// ```
#[proc_macro]
pub fn dialect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DialectInput);
    dialect::generate_dialect(input).unwrap_or_else(|error| error.to_compile_error())
}

#[proc_macro]
pub fn binary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    operation::generate_binary(set.dialect(), set.identifiers())
}

#[proc_macro]
pub fn unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    operation::generate_unary(set.dialect(), set.identifiers())
}

#[proc_macro]
pub fn typed_unary_operations(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as DialectOperationSet);

    operation::generate_typed_unary(set.dialect(), set.identifiers())
}

#[proc_macro]
pub fn type_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(r#type::generate(identifiers.identifiers()))
}

#[proc_macro]
pub fn attribute_check_functions(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(attribute::generate(identifiers.identifiers()))
}

#[proc_macro]
pub fn conversion_passes(stream: TokenStream) -> TokenStream {
    let identifiers = parse_macro_input!(stream as IdentifierList);

    convert_result(pass::generate(
        identifiers.identifiers(),
        "Conversion",
        |mut name| {
            name = name.strip_prefix("Convert").unwrap_or(name);
            name = name.strip_suffix("ConversionPass").unwrap_or(name);
            name.strip_suffix("Pass").unwrap_or(name).into()
        },
    ))
}

#[proc_macro]
pub fn passes(stream: TokenStream) -> TokenStream {
    let set = parse_macro_input!(stream as PassSet);

    convert_result(pass::generate(
        set.identifiers(),
        &set.prefix().value(),
        |name| name.into(),
    ))
}

fn convert_result(result: syn::Result<TokenStream>) -> TokenStream {
    result.unwrap_or_else(|error| error.to_compile_error().into())
}
