use super::SegmentKind;
use proc_macro2::{Ident, TokenStream};
use syn::Type;

pub trait OperationField {
    fn name(&self) -> &str;
    fn singular_identifier(&self) -> &Ident;
    fn plural_kind_identifier(&self) -> Ident;
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;
    fn is_optional(&self) -> bool;
    fn add_arguments(&self, name: &Ident) -> TokenStream;

    // The segment size array and slot recording this field's group length, for
    // operands and results of operations that size their groups by an
    // attribute. Required, so that every field states whether it has one.
    fn segment(&self) -> Option<(SegmentKind, usize)>;
}
