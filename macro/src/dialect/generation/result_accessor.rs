use super::element_accessor::generate_element_getter;
use crate::dialect::operation::OperationElement;
use proc_macro2::{Ident, Span, TokenStream};

pub fn generate_result_accessor(result: &OperationElement, length: usize) -> TokenStream {
    generate_element_getter(
        result,
        "result",
        "results",
        &Ident::new("ResultNotFound", Span::call_site()),
        length,
    )
}
