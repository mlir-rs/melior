use super::FieldKind;
use proc_macro2::Ident;

pub trait OperationField {
    fn name(&self) -> &str;
    fn kind(&self) -> FieldKind;
    fn index(&self) -> usize;
    fn singular_identifier(&self) -> &Ident;
    fn is_optional(&self) -> bool;

    // The slot in the segment size array recording this field's group length,
    // for operands and results of operations that size their groups by an
    // attribute. Required, so that every field states whether it has one.
    fn segment(&self) -> Option<usize>;
}
