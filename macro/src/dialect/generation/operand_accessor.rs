use super::element_accessor::generate_element_getter;
use crate::dialect::operation::Operand;
use proc_macro2::{Ident, Span, TokenStream};

pub fn generate_operand_accessor(operand: &Operand, length: usize) -> TokenStream {
    generate_element_getter(
        operand,
        "operand",
        "operands",
        &Ident::new("OperandNotFound", Span::call_site()),
        length,
    )
}
