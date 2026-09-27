use crate::dialect::{
    operation::{OperationElement, VariadicKind},
    utility::segment_size_attribute_name,
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub fn generate_element_getter(
    field: &impl OperationElement,
    singular_kind: &str,
    plural_kind: &str,
    error_variant: &Ident,
    index: usize,
    length: usize,
) -> TokenStream {
    let singular_kind_identifier = Ident::new(singular_kind, Span::call_site());
    let plural_kind_identifier = Ident::new(plural_kind, Span::call_site());
    let count = Ident::new(&format!("{singular_kind}_count"), Span::call_site());
    let name = field.name();

    let body = match field.variadic_kind() {
        VariadicKind::Simple { unfixed_seen } => {
            if field.is_optional() {
                // Optional element, and some singular elements.
                // Only present if the amount of groups is at least the number of
                // elements.
                quote! {
                    if self.operation.#count() < #length {
                        Err(::melior::Error::#error_variant(#name))
                    } else {
                        self.operation.#singular_kind_identifier(#index)
                    }
                }
            } else if field.is_variadic() {
                // A unfixed group
                // Length computed by subtracting the amount of other
                // singular elements from the number of elements.
                quote! {
                    let group_length = self.operation.#count() - #length + 1;
                    self.operation.#plural_kind_identifier().skip(#index).take(group_length)
                }
            } else if *unfixed_seen {
                // Single element after unfixed group
                // Compute the length of that variable group and take the next element
                quote! {
                    let group_length = self.operation.#count() - #length + 1;
                    self.operation.#singular_kind_identifier(#index + group_length - 1)
                }
            } else {
                // All elements so far are singular
                quote! {
                    self.operation.#singular_kind_identifier(#index)
                }
            }
        }
        VariadicKind::SameSize {
            unfixed_count,
            preceding_simple_count,
            preceding_variadic_count,
        } => {
            let get_elements = if field.is_unfixed() {
                quote! {
                    self.operation.#plural_kind_identifier().skip(start).take(group_len)
                }
            } else {
                quote! {
                    self.operation.#singular_kind_identifier(start)
                }
            };

            quote! {
                let total_var_len = self.operation.#count() - #unfixed_count + 1;
                let group_len = total_var_len / #unfixed_count;
                let start = #preceding_simple_count + #preceding_variadic_count * group_len;

                #get_elements
            }
        }
        VariadicKind::AttributeSized { segment_index } => {
            let segment_size_attribute = segment_size_attribute_name(singular_kind);
            let get_elements = if !field.is_unfixed() {
                quote! {
                    self.operation.#singular_kind_identifier(start)
                }
            } else if field.is_optional() {
                quote! {
                    if group_len == 0 {
                        Err(::melior::Error::#error_variant(#name))
                    } else {
                        self.operation.#singular_kind_identifier(start)
                    }
                }
            } else {
                quote! {
                    Ok(self.operation.#plural_kind_identifier().skip(start).take(group_len))
                }
            };

            quote! {
                let attribute =
                    ::melior::ir::attribute::DenseI32ArrayAttribute::<'c>::try_from(
                        self.operation
                        .attribute(#segment_size_attribute)?
                    )?;
                let start = (0..#segment_index)
                    .map(|index| attribute.element(index))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum::<i32>() as usize;
                let group_len = attribute.element(#segment_index)? as usize;

                #get_elements
            }
        }
    };

    let identifier = field.singular_identifier();
    let return_type = field.return_type();

    quote! {
        pub fn #identifier(&self) -> #return_type {
            #body
        }
    }
}
