use super::{FieldKind, OperationField};
use crate::dialect::{error::Error, utility::sanitize_snake_case_identifier};
use syn::Ident;

#[derive(Debug)]
pub struct Successor<'a> {
    name: &'a str,
    singular_identifier: Ident,
    variadic: bool,
    index: usize,
}

impl<'a> Successor<'a> {
    pub fn new(name: &'a str, variadic: bool, index: usize) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            variadic,
            index,
        })
    }

    pub const fn is_variadic(&self) -> bool {
        self.variadic
    }
}

impl OperationField for Successor<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn kind(&self) -> FieldKind {
        FieldKind::Successor
    }

    fn index(&self) -> usize {
        self.index
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn is_optional(&self) -> bool {
        false
    }

    // MLIR has no segment size attribute for successors.
    fn segment(&self) -> Option<usize> {
        None
    }
}
