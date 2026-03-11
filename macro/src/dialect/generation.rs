mod attribute_accessor;
mod element_accessor;
mod operand_accessor;
mod operation_builder;
mod region_accessor;
mod result_accessor;
mod successor_accessor;

use self::{
    attribute_accessor::generate_attribute_accessors,
    operand_accessor::generate_operand_accessor,
    operation_builder::{
        generate_default_constructor, generate_operation_builder, generate_operation_builder_fn,
    },
    region_accessor::generate_region_accessor,
    result_accessor::generate_result_accessor,
    successor_accessor::generate_successor_accessor,
};
use super::operation::{Operation, OperationBuilder};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation(operation: &Operation) -> TokenStream {
    let summary = operation.summary();
    let description = operation.description();
    let identifier = format_ident!("{}", operation.name());
    let operation_name = operation.full_operation_name();

    let result_accessors = operation
        .results()
        .enumerate()
        .map(|(index, result)| generate_result_accessor(result, index, operation.result_len()))
        .collect::<Vec<_>>();
    let operand_accessors = operation
        .operands()
        .enumerate()
        .map(|(index, operand)| generate_operand_accessor(operand, index, operation.operand_len()))
        .collect::<Vec<_>>();
    let region_accessors = operation
        .regions()
        .enumerate()
        .map(|(index, region)| generate_region_accessor(region, index))
        .collect::<Vec<_>>();
    let successor_accessors = operation
        .successors()
        .enumerate()
        .map(|(index, region)| generate_successor_accessor(region, index))
        .collect::<Vec<_>>();
    let attribute_accessors = operation
        .all_attributes()
        .map(generate_attribute_accessors)
        .collect::<Vec<_>>();

    let builder = OperationBuilder::new(operation);
    let builder_tokens = generate_operation_builder(&builder);
    let builder_fn = generate_operation_builder_fn(&builder);
    let default_constructor = generate_default_constructor(&builder);

    quote! {
        #[doc = #summary]
        #[doc = "\n\n"]
        #[doc = #description]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct #identifier<'c> {
            operation: ::melior::ir::operation::Operation<'c>,
        }

        impl<'c> #identifier<'c> {
            /// Returns a name.
            pub fn name() -> &'static str {
                #operation_name
            }

            /// Returns a generic operation.
            pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'c> {
                &self.operation
            }

            #builder_fn

            #(#result_accessors)*
            #(#operand_accessors)*
            #(#region_accessors)*
            #(#successor_accessors)*
            #(#attribute_accessors)*
        }

        #builder_tokens

        #default_constructor

        impl<'c> TryFrom<::melior::ir::operation::Operation<'c>> for #identifier<'c> {
            type Error = ::melior::Error;

            fn try_from(
                operation: ::melior::ir::operation::Operation<'c>,
                ) -> Result<Self, Self::Error> {
                let expected = #operation_name;
                let name = operation.name();

                if name.as_string_ref().as_str() == Ok(expected) {
                    Ok(Self { operation })
                } else {
                    Err(::melior::Error::OperationExpected {
                        expected,
                        actual: name.as_string_ref().as_str()
                            .unwrap_or("<non-utf8>").to_owned(),
                    })
                }
            }
        }

        impl<'c> From<#identifier<'c>> for ::melior::ir::operation::Operation<'c> {
            fn from(operation: #identifier<'c>) -> Self {
                operation.operation
            }
        }
    }
}
