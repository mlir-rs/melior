use crate::utility::map_name;
use convert_case::{Case, Casing};
use once_cell::sync::Lazy;
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use regex::{Captures, Regex};
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = map_name(
            &identifier
                .to_string()
                .strip_prefix("mlirTypeIsA")
                .unwrap()
                .to_case(Case::Snake),
        );

        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(" Returns `true` if a type is `{}`.", name);

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            fn #function_name(&self) -> bool {
                unsafe { mlir_sys::#identifier(self.to_raw()) }
            }
        }));
    }

    Ok(stream)
}
