use super::GeneratedField;
use crate::dialect::{
    operation::{ElementKind, OperationElement, OperationField, VariadicKind},
    utility::{generate_iterator_type, generate_result_type},
};
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Type, parse_quote};

impl GeneratedField for OperationElement<'_> {
    fn parameter_type(&self) -> Type {
        let r#type: Type = match self.element_kind() {
            ElementKind::Operand => parse_quote!(::melior::ir::Value<'c, '_>),
            ElementKind::Result => parse_quote!(::melior::ir::Type<'c>),
        };

        if self.is_variadic() {
            parse_quote! { &[#r#type] }
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = match self.element_kind() {
            ElementKind::Operand => parse_quote!(::melior::ir::Value<'c, '_>),
            ElementKind::Result => {
                parse_quote!(::melior::ir::operation::OperationResult<'c, '_>)
            }
        };

        if !self.is_variadic() {
            generate_result_type(r#type)
        } else if self.position().variadic_kind() == Some(VariadicKind::AttributeSized) {
            generate_result_type(generate_iterator_type(r#type))
        } else {
            generate_iterator_type(r#type)
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
