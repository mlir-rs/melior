mod attribute;
mod operation_element;
mod region;
mod successor;

use crate::dialect::operation::{FieldKind, OperationField};
use proc_macro2::{Ident, TokenStream};
use quote::format_ident;
use syn::Type;

pub trait GeneratedField: OperationField {
    fn parameter_type(&self) -> Type;
    fn return_type(&self) -> Type;

    fn add_identifier(&self) -> Ident {
        generate_add_identifier(self.kind())
    }

    fn add_arguments(&self) -> TokenStream;
}

fn generate_add_identifier(kind: FieldKind) -> Ident {
    format_ident!("add_{}", kind.plural_name())
}
