use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use std::error::Error;

const CREATE_FUNCTION_PREFIX: &str = "mlirCreate";

pub fn generate(
    names: &[Ident],
    extract_pass_name: impl Fn(&str) -> String,
) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for name in names {
        let foreign_name = name.to_string();
        let foreign_name = foreign_name.strip_prefix(CREATE_FUNCTION_PREFIX).unwrap();
        let pass_name = extract_pass_name(foreign_name);

        let function_name = create_function_name("create", &pass_name, name.span());
        let document = format!(" Creates a `{pass_name}` pass.");

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            pub fn #function_name() -> crate::pass::Pass {
                unsafe { crate::pass::Pass::__private_from_raw_fn(mlir_sys::#name) }
            }
        }));

        let foreign_function_name =
            Ident::new(&("mlirRegister".to_owned() + foreign_name), name.span());
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
        &format!("{}_{}", prefix, &pass_name.to_case(Case::Snake)),
        span,
    )
}
