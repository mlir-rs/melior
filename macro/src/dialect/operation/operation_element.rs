use super::{ElementPosition, FieldKind, OperationField, VariadicKind};
use crate::dialect::{
    error::Error,
    r#type::Type as ElementType,
    utility::{generate_iterator_type, generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{Type, parse_quote};

#[derive(Debug)]
pub struct OperationElement<'a> {
    name: &'a str,
    kind: FieldKind,
    singular_identifier: Ident,
    r#type: ElementType,
    position: ElementPosition,
}

impl<'a> OperationElement<'a> {
    pub fn new(
        name: &'a str,
        kind: FieldKind,
        r#type: ElementType,
        position: ElementPosition,
    ) -> Result<Self, Error> {
        Ok(Self {
            name,
            kind,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            r#type,
            position,
        })
    }

    pub const fn is_variadic(&self) -> bool {
        self.r#type.is_variadic()
    }

    pub const fn is_unfixed(&self) -> bool {
        self.r#type.is_unfixed()
    }

    pub const fn position(&self) -> ElementPosition {
        self.position
    }
}

impl OperationField for OperationElement<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn kind(&self) -> FieldKind {
        self.kind
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn parameter_type(&self) -> Type {
        let r#type: Type = match self.kind {
            FieldKind::Operand => parse_quote!(::melior::ir::Value<'c, '_>),
            FieldKind::Result => parse_quote!(::melior::ir::Type<'c>),
            FieldKind::Region | FieldKind::Successor | FieldKind::Attribute => unreachable!(),
        };

        if self.is_variadic() {
            parse_quote! { &[#r#type] }
        } else {
            r#type
        }
    }

    fn return_type(&self) -> Type {
        let r#type: Type = match self.kind {
            FieldKind::Operand => parse_quote!(::melior::ir::Value<'c, '_>),
            FieldKind::Result => {
                parse_quote!(::melior::ir::operation::OperationResult<'c, '_>)
            }
            FieldKind::Region | FieldKind::Successor | FieldKind::Attribute => unreachable!(),
        };

        if !self.is_variadic() {
            generate_result_type(r#type)
        } else if self.position.variadic_kind() == Some(VariadicKind::AttributeSized) {
            generate_result_type(generate_iterator_type(r#type))
        } else {
            generate_iterator_type(r#type)
        }
    }

    fn is_optional(&self) -> bool {
        self.r#type.is_optional()
    }

    fn add_arguments(&self) -> TokenStream {
        let name = &self.singular_identifier;

        if self.is_variadic() {
            quote! { #name }
        } else {
            quote! { &[#name] }
        }
    }

    fn segment(&self) -> Option<usize> {
        (self.position.variadic_kind() == Some(VariadicKind::AttributeSized))
            .then_some(self.position.index())
    }
}
