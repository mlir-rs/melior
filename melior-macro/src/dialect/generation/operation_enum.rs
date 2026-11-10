use crate::dialect::{operation::Operation, utility::capitalize_string};
use convert_case::{Case, Casing};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation_enum(
    dialect_name: &str,
    operations: &[Operation],
) -> Option<TokenStream> {
    let enum_identifier = format_ident!("{}Operation", dialect_name.to_case(Case::Pascal));
    let identifiers = operations
        .iter()
        .map(|operation| format_ident!("{}", operation.name()))
        .collect::<Vec<_>>();
    let variants = operations
        .iter()
        .map(|operation| format_ident!("{}", operation.short_name()))
        .collect::<Vec<_>>();
    let documents = operations
        .iter()
        .map(|operation| format!("{}.", capitalize_string(&operation.documentation_name())));
    let full_names = operations.iter().map(Operation::full_operation_name);

    (!operations.is_empty()).then(|| {
        quote! {
            /// An operation in this dialect.
            #[derive(Clone, Debug, PartialEq, Eq)]
            pub enum #enum_identifier<'b> {
                #(
                    #[doc = #documents]
                    #variants(#identifiers<'b>)
                ),*
            }

            impl<'b> std::fmt::Display for #enum_identifier<'b> {
                fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                   std::fmt::Display::fmt(self.as_operation(), formatter)
                }
            }

            impl<'b> #enum_identifier<'b> {
                /// Converts a generic operation into one in this dialect, or returns it as an
                /// error if it is not in this dialect.
                pub fn try_new(operation: ::melior::ir::operation::Operation<'b>) -> Result<Self, ::melior::ir::operation::Operation<'b>> {
                    let name = operation.name();
                    let Ok(name_str) = name.as_string_ref().as_str() else {
                        return Err(operation);
                    };
                    match name_str {
                        #(
                            #full_names => Ok(
                                #enum_identifier::#variants(
                                    #identifiers::try_from(operation)
                                        .expect("operation should match type"),
                                ),
                            ),
                        )*
                        _ => Err(operation),
                    }
                }
            }

            impl<'b> #enum_identifier<'b> {
                /// Returns a generic operation.
                pub fn as_operation(&self) -> &::melior::ir::operation::Operation<'b> {
                    match self {
                        #(#enum_identifier::#variants(operation) => operation.as_operation(),)*
                    }
                }
            }

            #(
                impl<'b> From<#identifiers<'b>> for #enum_identifier<'b> {
                    fn from(operation: #identifiers<'b>) -> Self {
                        #enum_identifier::#variants(operation)
                    }
                }
            )*
        }
    })
}
