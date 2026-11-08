use super::FieldKind;
use proc_macro2::{Ident, TokenStream};
use syn::Type;

pub trait OperationField {
    fn name(&self) -> &str;
    fn kind(&self) -> FieldKind;
    fn index(&self) -> usize;
    fn singular_identifier(&self) -> &Ident;
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;
    fn is_optional(&self) -> bool;

    fn add_identifier(&self) -> Ident {
        self.kind().add_identifier()
    }

    fn add_arguments(&self) -> TokenStream;

    // The slot in the segment size array recording this field's group length,
    // for operands and results of operations that size their groups by an
    // attribute. Required, so that every field states whether it has one.
    fn segment(&self) -> Option<usize>;
}
