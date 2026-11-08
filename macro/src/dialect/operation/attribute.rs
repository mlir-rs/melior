use crate::dialect::{
    error::Error,
    operation::{FieldKind, operation_field::OperationField},
    utility::{generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Ident, Type, parse_quote};
use tblgen::{Record, error::TableGenError};

#[derive(Debug)]
pub struct Attribute<'a> {
    name: &'a str,
    singular_identifier: Ident,
    set_identifier: Ident,
    remove_identifier: Ident,
    storage_type: Ident,
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
            storage_type: Ident::new(
                Self::build_storage_type(
                    // TODO Handle `?` attribute initializers properly.
                    record
                        .string_value("storageType")
                        .unwrap_or_default()
                        .trim(),
                ),
                Span::call_site(),
            ),
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

    pub fn is_type(&self) -> bool {
        self.storage_type == "TypeAttribute"
    }

    fn build_storage_type(storage_type: &str) -> &'static str {
        match storage_type {
            "::mlir::ArrayAttr" => "ArrayAttribute",
            "::mlir::BoolAttr" => "BoolAttribute",
            "::mlir::DenseElementsAttr" => "DenseElementsAttribute",
            "::mlir::DenseI32ArrayAttr" => "DenseI32ArrayAttribute",
            "::mlir::DenseI64ArrayAttr" => "DenseI64ArrayAttribute",
            "::mlir::DictionaryAttr" => "DictionaryAttribute",
            "::mlir::FlatSymbolRefAttr" => "FlatSymbolRefAttribute",
            "::mlir::FloatAttr" => "FloatAttribute",
            "::mlir::IntegerAttr" => "IntegerAttribute",
            "::mlir::StringAttr" => "StringAttribute",
            "::mlir::StridedLayoutAttr" => "StridedLayoutAttribute",
            "::mlir::TypeAttr" => "TypeAttribute",
            _ => "Attribute",
        }
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

    fn parameter_type(&self) -> Type {
        let r#type = &self.storage_type;

        parse_quote!(::melior::ir::attribute::#r#type<'c>)
    }

    fn return_type(&self) -> Type {
        generate_result_type(self.parameter_type())
    }

    fn is_optional(&self) -> bool {
        self.optional || self.default
    }

    fn add_arguments(&self) -> TokenStream {
        let name = &self.singular_identifier;
        let name_string = &self.name;

        quote! {
            &[(
                ::melior::ir::Identifier::new(self.context, #name_string),
                #name.into(),
            )]
        }
    }

    // Attributes are not operands or results, so they have no group to size.
    fn segment(&self) -> Option<usize> {
        None
    }
}
