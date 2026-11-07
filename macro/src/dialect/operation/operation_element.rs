use super::{ElementKind, ElementPosition, FieldKind, OperationField, VariadicKind};
use crate::dialect::{
    error::Error, r#type::Type as ElementType, utility::sanitize_snake_case_identifier,
};
use proc_macro2::Ident;

#[derive(Debug)]
pub struct OperationElement<'a> {
    name: &'a str,
    kind: ElementKind,
    singular_identifier: Ident,
    r#type: ElementType,
    position: ElementPosition,
}

impl<'a> OperationElement<'a> {
    pub fn new(
        name: &'a str,
        kind: ElementKind,
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

    pub const fn element_kind(&self) -> ElementKind {
        self.kind
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
        self.kind.into()
    }

    fn index(&self) -> usize {
        self.position.index()
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn is_optional(&self) -> bool {
        self.r#type.is_optional()
    }

    fn segment(&self) -> Option<usize> {
        (self.position.variadic_kind() == Some(VariadicKind::AttributeSized))
            .then_some(self.position.index())
    }
}
