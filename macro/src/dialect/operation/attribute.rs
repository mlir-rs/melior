use crate::dialect::{
    error::Error,
    operation::{FieldKind, operation_field::OperationField},
    utility::sanitize_snake_case_identifier,
};
use syn::Ident;
use tblgen::{Record, error::TableGenError};

#[derive(Debug)]
pub struct Attribute<'a> {
    name: &'a str,
    singular_identifier: Ident,
    set_identifier: Ident,
    remove_identifier: Ident,
    storage_type: String,
    optional: bool,
    default: bool,
    index: usize,
}

impl<'a> Attribute<'a> {
    pub fn new(name: &'a str, record: Record<'a>, index: usize) -> Result<Self, Error> {
        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            set_identifier: sanitize_snake_case_identifier(&format!("set_{name}"))?,
            remove_identifier: sanitize_snake_case_identifier(&format!("remove_{name}"))?,
            // TODO Handle `?` attribute initializers properly.
            storage_type: record
                .string_value("storageType")
                .unwrap_or_default()
                .trim()
                .into(),
            optional: record.bit_value("isOptional")?,
            default: match record.string_value("defaultValue") {
                Ok(value) => !value.is_empty(),
                Err(error) => {
                    // `defaultValue` can be uninitialized.
                    if !matches!(error.error(), TableGenError::InitConversion { .. }) {
                        return Err(error.into());
                    }

                    false
                }
            },
            index,
        })
    }

    pub const fn set_identifier(&self) -> &Ident {
        &self.set_identifier
    }

    pub const fn remove_identifier(&self) -> &Ident {
        &self.remove_identifier
    }

    pub fn storage_type(&self) -> &str {
        &self.storage_type
    }

    pub fn is_type(&self) -> bool {
        self.storage_type == "::mlir::TypeAttr"
    }
}

impl OperationField for Attribute<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn kind(&self) -> FieldKind {
        FieldKind::Attribute
    }

    fn index(&self) -> usize {
        self.index
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn is_optional(&self) -> bool {
        self.optional || self.default
    }

    // Attributes are not operands or results, so they have no group to size.
    fn segment(&self) -> Option<usize> {
        None
    }
}
