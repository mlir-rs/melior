mod builder;
mod ordered_field;
mod type_argument;

pub use self::{
    builder::TypeStateBuilder, ordered_field::OrderedField, type_argument::TypeArgument,
};
use crate::dialect::operation::FieldKind;
use quote::format_ident;
use syn::Ident;

#[derive(Debug)]
struct Slot {
    kind: FieldKind,
    index: usize,
    parameter: Ident,
    ordered: bool,
    optional: bool,
}

impl Slot {
    fn final_argument(&self) -> TypeArgument {
        if self.optional {
            TypeArgument::Parameter(self.parameter.clone())
        } else {
            TypeArgument::Set
        }
    }
}

#[derive(Debug)]
pub struct TypeState {
    slots: Vec<Slot>,
}

impl TypeState {
    pub fn new(
        results: Vec<OrderedField>,
        operands: Vec<OrderedField>,
        regions: Vec<OrderedField>,
        successors: Vec<OrderedField>,
        attributes: Vec<usize>,
    ) -> Self {
        Self {
            slots: [
                (FieldKind::Result, results),
                (FieldKind::Operand, operands),
                (FieldKind::Region, regions),
                (FieldKind::Successor, successors),
            ]
            .into_iter()
            .flat_map(|(kind, fields)| {
                fields
                    .into_iter()
                    .enumerate()
                    .map(move |(index, field)| Slot {
                        kind,
                        index,
                        parameter: format_ident!("{}{index}", kind.type_parameter_prefix()),
                        ordered: true,
                        optional: field.is_optional(),
                    })
            })
            .chain(
                attributes
                    .into_iter()
                    .enumerate()
                    .map(|(position, index)| Slot {
                        kind: FieldKind::Attribute,
                        index,
                        parameter: format_ident!(
                            "{}{position}",
                            FieldKind::Attribute.type_parameter_prefix()
                        ),
                        ordered: false,
                        optional: false,
                    }),
            )
            .collect(),
        }
    }

    pub fn has_parameter(&self, kind: FieldKind, index: usize) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.kind == kind && slot.index == index)
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Ident> {
        self.slots.iter().map(|slot| &slot.parameter)
    }

    pub fn initial_arguments(&self) -> impl Iterator<Item = TypeArgument> + '_ {
        self.slots.iter().map(|_| TypeArgument::Unset)
    }

    pub fn arguments_with(
        &self,
        kind: FieldKind,
        index: usize,
        set: bool,
    ) -> impl Iterator<Item = TypeArgument> + '_ {
        debug_assert!(self.has_parameter(kind, index));

        self.slots.iter().map(move |slot| {
            if slot.kind != kind || (!slot.ordered && slot.index != index) {
                TypeArgument::Parameter(slot.parameter.clone())
            } else if slot.index == index {
                if set {
                    TypeArgument::Set
                } else {
                    TypeArgument::Unset
                }
            } else if slot.index < index {
                slot.final_argument()
            } else {
                TypeArgument::Unset
            }
        })
    }

    pub fn final_arguments(&self) -> impl Iterator<Item = TypeArgument> + '_ {
        self.slots.iter().map(Slot::final_argument)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn create_type_state() -> TypeState {
        TypeState::new(
            vec![OrderedField::new(true), OrderedField::new(false)],
            vec![OrderedField::new(false)],
            vec![],
            vec![],
            vec![0, 2],
        )
    }

    fn parameter(name: &str) -> TypeArgument {
        TypeArgument::Parameter(format_ident!("{name}"))
    }

    fn parameters(arguments: impl Iterator<Item = TypeArgument>) -> Vec<String> {
        arguments
            .filter_map(|argument| argument.parameter().map(ToString::to_string))
            .collect()
    }

    #[test]
    fn check_parameter() {
        let state = create_type_state();

        assert!(state.has_parameter(FieldKind::Result, 1));
        assert!(state.has_parameter(FieldKind::Operand, 0));
        assert!(!state.has_parameter(FieldKind::Operand, 1));
        assert!(!state.has_parameter(FieldKind::Region, 0));
        assert!(state.has_parameter(FieldKind::Attribute, 2));
        assert!(!state.has_parameter(FieldKind::Attribute, 1));
    }

    #[test]
    fn build_parameters() {
        assert_eq!(
            create_type_state()
                .parameters()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["T0", "T1", "O0", "A0", "A1"]
        );
    }

    #[test]
    fn build_initial_arguments() {
        assert_eq!(
            create_type_state().initial_arguments().collect::<Vec<_>>(),
            [
                TypeArgument::Unset,
                TypeArgument::Unset,
                TypeArgument::Unset,
                TypeArgument::Unset,
                TypeArgument::Unset
            ]
        );
    }

    #[test]
    fn build_arguments_with_ordered_field() {
        assert_eq!(
            create_type_state()
                .arguments_with(FieldKind::Result, 1, true)
                .collect::<Vec<_>>(),
            [
                parameter("T0"),
                TypeArgument::Set,
                parameter("O0"),
                parameter("A0"),
                parameter("A1")
            ]
        );
        assert_eq!(
            create_type_state()
                .arguments_with(FieldKind::Result, 0, false)
                .collect::<Vec<_>>(),
            [
                TypeArgument::Unset,
                TypeArgument::Unset,
                parameter("O0"),
                parameter("A0"),
                parameter("A1")
            ]
        );
    }

    #[test]
    fn build_arguments_with_attribute() {
        assert_eq!(
            create_type_state()
                .arguments_with(FieldKind::Attribute, 2, true)
                .collect::<Vec<_>>(),
            [
                parameter("T0"),
                parameter("T1"),
                parameter("O0"),
                parameter("A0"),
                TypeArgument::Set
            ]
        );
    }

    #[test]
    fn build_final_arguments() {
        assert_eq!(
            create_type_state().final_arguments().collect::<Vec<_>>(),
            [
                parameter("T0"),
                TypeArgument::Set,
                TypeArgument::Set,
                TypeArgument::Set,
                TypeArgument::Set
            ]
        );
    }

    #[test]
    fn derive_parameters_from_arguments() {
        let state = create_type_state();

        assert_eq!(
            parameters(state.arguments_with(FieldKind::Result, 1, false)),
            ["T0", "O0", "A0", "A1"]
        );
        assert_eq!(
            parameters(state.arguments_with(FieldKind::Attribute, 2, false)),
            ["T0", "T1", "O0", "A0"]
        );
        assert_eq!(parameters(state.final_arguments()), ["T0"]);
        assert_eq!(parameters(state.initial_arguments()), [] as [&str; 0]);
    }

    #[test]
    fn distinguish_fields_of_different_kinds() {
        assert_eq!(
            create_type_state()
                .arguments_with(FieldKind::Operand, 0, true)
                .collect::<Vec<_>>(),
            [
                parameter("T0"),
                parameter("T1"),
                TypeArgument::Set,
                parameter("A0"),
                parameter("A1")
            ]
        );
    }

    #[test]
    #[should_panic(expected = "has_parameter")]
    fn fail_to_build_arguments_with_field_past_end() {
        let _ = create_type_state().arguments_with(FieldKind::Result, 2, true);
    }

    #[test]
    #[should_panic(expected = "has_parameter")]
    fn fail_to_build_arguments_with_attribute_without_parameter() {
        let _ = create_type_state().arguments_with(FieldKind::Attribute, 1, true);
    }
}
