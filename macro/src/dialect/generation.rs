mod attribute_accessor;
mod field_accessor;
mod operation_builder;
mod region_accessor;
mod successor_accessor;

use self::{
    attribute_accessor::generate_attribute_accessors,
    field_accessor::generate_accessor,
    operation_builder::{
        generate_default_constructor, generate_operation_builder, generate_operation_builder_fn,
    },
    region_accessor::generate_region_accessor,
    successor_accessor::generate_successor_accessor,
};
use super::operation::{Operation, OperationBuilder};
use crate::dialect::error::Error;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation(operation: &Operation) -> Result<TokenStream, Error> {
    let summary = operation.summary()?;
    let description = operation.description()?;
    let class_name = format_ident!("{}", operation.class_name()?);
    let name = &operation.full_name()?;

    let field_accessors = operation
        .general_fields()
        .map(generate_accessor)
        .collect::<Result<Vec<_>, _>>()?;
    let successor_accessors = operation
        .successors()
        .enumerate()
        .map(|(index, region)| generate_successor_accessor(index, region))
        .collect::<Vec<_>>();
    let region_accessors = operation
        .regions()
        .enumerate()
        .map(|(index, region)| generate_region_accessor(index, region))
        .collect::<Vec<_>>();
    let attribute_accessors = operation
        .attributes()
        .map(generate_attribute_accessors)
        .collect::<Result<Vec<_>, _>>()?;

    let builder = OperationBuilder::new(operation)?;
    let builder_tokens = generate_operation_builder(&builder)?;
    let builder_fn = generate_operation_builder_fn(&builder);
    let default_constructor = generate_default_constructor(&builder)?;

    Ok(quote! {
        #[doc = #summary]
        #[doc = "\n\n"]
        #[doc = #description]
        pub struct #class_name<'c> {
            operation: ::melior::ir::operation::Operation<'c>,
        }

        impl<'c> #class_name<'c> {
            pub fn name() -> &'static str {
                #name
            }

            pub fn operation(&self) -> &::melior::ir::operation::Operation<'c> {
                &self.operation
            }

            #builder_fn

            #(#field_accessors)*
            #(#successor_accessors)*
            #(#region_accessors)*
            #(#attribute_accessors)*
        }

        #builder_tokens

        #default_constructor

        impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for #class_name<'c> {
            type Error = ::melior::Error;

            fn try_from(
                operation: ::melior::ir::operation::Operation<'c>,
            ) -> Result<Self, Self::Error> {
                // TODO Check an operation name.
                Ok(Self { operation })
            }
        }

        impl<'c> From<#class_name<'c>> for ::melior::ir::operation::Operation<'c> {
            fn from(operation: #class_name<'c>) -> ::melior::ir::operation::Operation<'c> {
                operation.operation
            }
        }
    })
}
