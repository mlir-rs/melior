use super::{GeneratedField, generate_add_identifier};
use crate::dialect::{
    operation::{OperationField, Region},
    utility::{generate_iterator_type, generate_result_type},
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{Type, parse_quote};

impl GeneratedField for Region<'_> {
    fn parameter_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::Region<'c>);

        if self.is_variadic() {
            parse_quote!(Vec<#r#type>)
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::RegionRef<'c, '_>);

        if self.is_variadic() {
            generate_iterator_type(r#type)
        } else {
            generate_result_type(r#type)
        }
    }

    fn add_identifier(&self) -> Ident {
        let identifier = generate_add_identifier(self.kind());

        if self.is_variadic() {
            format_ident!("{identifier}_vec")
        } else {
            identifier
        }
    }

    fn add_arguments(&self) -> TokenStream {
        let name = self.singular_identifier();

        if self.is_variadic() {
            quote! { #name }
        } else {
            quote! { [#name] }
        }
    }
}
