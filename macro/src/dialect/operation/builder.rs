mod type_state;

use self::type_state::{OrderedField, TypeState};
use super::{Operation, OperationField};
use quote::format_ident;
use syn::Ident;

pub struct OperationBuilder<'a> {
    operation: &'a Operation<'a>,
    identifier: Ident,
    type_state: TypeState,
}

impl<'a> OperationBuilder<'a> {
    pub fn new(operation: &'a Operation<'a>) -> Self {
        Self {
            operation,
            identifier: format_ident!("{}Builder", operation.name()),
            type_state: Self::create_type_state(operation),
        }
    }

    pub const fn operation(&self) -> &Operation<'_> {
        self.operation
    }

    pub const fn identifier(&self) -> &Ident {
        &self.identifier
    }

    pub const fn type_state(&self) -> &TypeState {
        &self.type_state
    }

    fn create_type_state(operation: &Operation) -> TypeState {
        TypeState::new(
            Self::build_ordered_fields(operation.non_inferred_results()),
            Self::build_ordered_fields(operation.operands()),
            Self::build_ordered_fields(operation.regions()),
            Self::build_ordered_fields(operation.successors()),
            Self::build_names(operation.required_attributes()),
        )
    }

    fn build_ordered_fields<'b>(
        fields: impl Iterator<Item = &'b (impl OperationField + 'b)>,
    ) -> Vec<OrderedField> {
        fields
            .map(|field| OrderedField::new(field.name().into(), field.is_optional()))
            .collect()
    }

    fn build_names<'b>(
        fields: impl Iterator<Item = &'b (impl OperationField + 'b)>,
    ) -> Vec<String> {
        fields.map(|field| field.name().into()).collect()
    }
}
