use super::generated_field::GeneratedField;
use crate::dialect::operation::{OperationField, Successor};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_successor_accessor(successor: &Successor) -> TokenStream {
    let identifier = successor.singular_identifier();
    let return_type = successor.return_type();
    let index = successor.index();
    let body = if successor.is_variadic() {
        // Only the last successor can be variadic.
        quote! {
            self.operation.successors().skip(#index)
        }
    } else {
        quote! {
            self.operation.successor(#index)
        }
    };
    let doc = format!(
        "Returns the `{}` {}.",
        successor.name(),
        if successor.is_variadic() {
            "successors"
        } else {
            "successor"
        }
    );

    quote! {
        #[doc = #doc]
        pub fn #identifier(&self) -> #return_type {
            #body
        }
    }
}
