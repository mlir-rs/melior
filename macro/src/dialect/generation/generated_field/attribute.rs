use super::GeneratedField;
use crate::dialect::{
    operation::{Attribute, OperationField},
    utility::generate_result_type,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{Type, parse_quote};

impl GeneratedField for Attribute<'_> {
    fn parameter_type(&self) -> Type {
        let r#type = Ident::new(build_storage_type(self.storage_type()), Span::call_site());

        parse_quote!(::melior::ir::attribute::#r#type<'c>)
    }

    fn return_type(&self) -> Type {
        generate_result_type(self.parameter_type())
    }

    fn add_arguments(&self) -> TokenStream {
        let name = self.singular_identifier();
        let name_string = self.name();

        quote! {
            &[(
                ::melior::ir::Identifier::new(self.context, #name_string),
                #name.into(),
            )]
        }
    }
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
