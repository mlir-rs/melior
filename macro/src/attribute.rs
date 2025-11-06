use crate::utility::map_name;
use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;
use std::error::Error;

pub fn generate(identifiers: &[Ident]) -> Result<TokenStream, Box<dyn Error>> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = map_name(
            &identifier
                .to_string()
                .strip_prefix("mlirAttributeIsA")
                .unwrap()
                .to_case(Case::Snake)
                .trim_end_matches('_'),
        );

        let function_name = Ident::new(&format!("is_{}", &name), identifier.span());
        let document = format!(
            " Returns `true` if an attribute is {}.",
            name.replace('_', " ")
        );

        stream.extend(TokenStream::from(quote! {
            #[doc = #document]
            fn #function_name(&self) -> bool {
                unsafe { mlir_sys::#identifier(self.to_raw()) }
            }
        }));
    }

    Ok(stream)
}
