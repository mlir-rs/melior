use super::GeneratedField;
use crate::dialect::{
    operation::{OperationField, Successor},
    utility::{generate_iterator_type, generate_result_type},
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Type, parse_quote};

impl GeneratedField for Successor<'_> {
    fn parameter_type(&self) -> Type {
        let r#type: Type = parse_quote!(&::melior::ir::Block<'c>);

        if self.is_variadic() {
            parse_quote!(&[#r#type])
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::BlockRef<'c, '_>);

        if self.is_variadic() {
            generate_iterator_type(r#type)
        } else {
            generate_result_type(r#type)
        }
    }

    fn add_arguments(&self) -> TokenStream {
        let name = self.singular_identifier();

        if self.is_variadic() {
            quote! { #name }
        } else {
            quote! { &[#name] }
        }
    }
}
