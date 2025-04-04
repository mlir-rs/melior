use crate::dialect::{
    error::Error,
    operation::operation_field::OperationField,
    utility::{generate_result_type, sanitize_snake_case_identifier},
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::{collections::HashMap, sync::LazyLock};
use syn::{parse_quote, Ident, Type};
use tblgen::{error::TableGenError, Record};

macro_rules! prefixed_string {
    ($prefix:literal, $name:ident) => {
        concat!($prefix, stringify!($name))
    };
}

macro_rules! mlir_attribute {
    ($name:ident) => {
        prefixed_string!("::mlir::", $name)
    };
}

macro_rules! melior_attribute {
    ($name:ident) => {
        prefixed_string!("::melior::ir::attribute::", $name)
    };
}

static ATTRIBUTE_TYPES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut map = HashMap::new();

    macro_rules! initialize_attributes {
        ($($mlir:ident => $melior:ident),* $(,)*) => {
            $(
                map.insert(
                    mlir_attribute!($mlir),
                    melior_attribute!($melior),
                );
            )*
        };
    }

    initialize_attributes!(
        ArrayAttr => ArrayAttribute,
        Attribute => Attribute,
        DenseElementsAttr => DenseElementsAttribute,
        DenseI32ArrayAttr => DenseI32ArrayAttribute,
        FlatSymbolRefAttr => FlatSymbolRefAttribute,
        FloatAttr => FloatAttribute,
        IntegerAttr => IntegerAttribute,
        StringAttr => StringAttribute,
        TypeAttr => TypeAttribute,
    );

    map
});

#[derive(Debug)]
pub struct Attribute<'a> {
    name: &'a str,
    singular_identifier: Ident,
    storage_type_string: String,
    set_identifier: Ident,
    remove_identifier: Ident,
    storage_type: Type,
    optional: bool,
    default: bool,
}

impl<'a> Attribute<'a> {
    pub fn new(name: &'a str, record: Record<'a>) -> Result<Self, Error> {
        let storage_type_string = record.string_value("storageType")?;

        Ok(Self {
            name,
            singular_identifier: sanitize_snake_case_identifier(name)?,
            set_identifier: sanitize_snake_case_identifier(&format!("set_{name}"))?,
            remove_identifier: sanitize_snake_case_identifier(&format!("remove_{name}"))?,
            storage_type: syn::parse_str(
                ATTRIBUTE_TYPES
                    .get(storage_type_string.trim())
                    .copied()
                    .unwrap_or(melior_attribute!(Attribute)),
            )?,
            storage_type_string,
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
        })
    }

    pub const fn set_identifier(&self) -> &Ident {
        &self.set_identifier
    }

    pub const fn remove_identifier(&self) -> &Ident {
        &self.remove_identifier
    }

    pub fn is_unit(&self) -> bool {
        self.storage_type_string == mlir_attribute!(UnitAttr)
    }
}

impl OperationField for Attribute<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn singular_identifier(&self) -> &Ident {
        &self.singular_identifier
    }

    fn plural_kind_identifier(&self) -> Ident {
        Ident::new("attributes", Span::call_site())
    }

    fn parameter_type(&self) -> Type {
        if self.is_unit() {
            parse_quote!(bool)
        } else {
            let r#type = &self.storage_type;
            parse_quote!(#r#type<'c>)
        }
    }

    fn return_type(&self) -> Type {
        if self.is_unit() {
            parse_quote!(bool)
        } else {
            generate_result_type(self.parameter_type())
        }
    }

    fn is_optional(&self) -> bool {
        self.optional || self.default
    }

    fn add_arguments(&self, name: &Ident) -> TokenStream {
        let name_string = &self.name;

        quote! {
            &[(
                ::melior::ir::Identifier::new(self.context, #name_string),
                #name.into(),
            )]
        }
    }
}
