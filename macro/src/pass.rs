use crate::utility::strip_prefix;
use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;

const CREATE_FUNCTION_PREFIX: &str = "mlirCreate";

pub fn generate(
    names: &[Ident],
    prefix: &str,
    extract_pass_name: impl Fn(&str) -> String,
) -> syn::Result<TokenStream> {
    let mut stream = TokenStream::new();

    for name in names {
        let suffix = strip_prefix(name, &format!("{CREATE_FUNCTION_PREFIX}{prefix}"))?;
        let pass_name = extract_pass_name(&suffix);

        let function_name = create_function_name("create", &pass_name, name.span());
        let document = format!(" Creates a `{pass_name}` pass.");

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                unsafe { crate::pass::Pass::__private_from_raw_fn(mlir_sys::#name) }
            }
        }));

        let foreign_function_name =
            Ident::new(&format!("mlirRegister{prefix}{suffix}"), name.span());
        let function_name = create_function_name("register", &pass_name, name.span());
        let document = format!(" Registers a `{pass_name}` pass.");

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() {
                unsafe { mlir_sys::#foreign_function_name() }
            }
        }));
    }

    Ok(stream)
}

fn create_function_name(prefix: &str, pass_name: &str, span: Span) -> Ident {
    Ident::new(
        &format!("{}_{}", prefix, pass_name.to_case(Case::Snake)),
        span,
    )
}
