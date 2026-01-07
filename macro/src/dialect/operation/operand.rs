use super::{OperationElement, OperationField, VariadicKind};
use crate::dialect::{
    error::Error,
    r#type::Type as ElementType,
    utility::{generate_iterator_type, generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{Type, parse_quote};

#[derive(Debug)]
pub struct Operand<'a> {
    name: &'a str,
    singular_identifier: Ident,
    r#type: ElementType,
    variadic_kind: VariadicKind,
}

impl<'a> Operand<'a> {
    pub fn new(
        name: &'a str,
        r#type: ElementType,
        variadic_kind: VariadicKind,
    ) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            r#type,
            variadic_kind,
        })
    }
}

impl OperationField for Operand<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn plural_kind_identifier(&self) -> Ident {
        format_ident!("operands")
    }

    fn parameter_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::Value<'c, '_>);

        if self.r#type.is_variadic() {
            parse_quote! { &[#r#type] }
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = parse_quote!(::melior::ir::Value<'c, '_>);

        if !self.r#type.is_variadic() {
            generate_result_type(r#type)
        } else if self.variadic_kind == VariadicKind::AttributeSized {
            generate_result_type(generate_iterator_type(r#type))
        } else {
            generate_iterator_type(r#type)
        }
    }

    fn is_optional(&self) -> bool {
        self.r#type.is_optional()
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        if self.r#type.is_variadic() {
            quote! { #name }
        } else {
            quote! { &[#name] }
        }
    }
}

impl OperationElement for Operand<'_> {
    fn is_variadic(&self) -> bool {
        self.r#type.is_variadic()
    }

    fn variadic_kind(&self) -> &VariadicKind {
        &self.variadic_kind
    }
}
