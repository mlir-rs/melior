use crate::utility::{map_name, strip_prefix};
use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::quote;

pub fn generate(identifiers: &[Ident]) -> syn::Result<TokenStream> {
    let mut stream = TokenStream::new();

    for identifier in identifiers {
        let name = map_name(&strip_prefix(identifier, "mlirAttributeIsA")?.to_case(Case::Snake));

        let function_name = Ident::new(&format!("is_{}", name), identifier.span());
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
