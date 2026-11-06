mod ordered_field;

pub use self::ordered_field::OrderedField;
use crate::dialect::operation::FieldKind;
use core::iter::repeat_n;
use quote::format_ident;
use syn::{GenericArgument, parse_quote};

#[derive(Debug)]
pub struct TypeState {
    results: Vec<OrderedField>,
    operands: Vec<OrderedField>,
    regions: Vec<OrderedField>,
    successors: Vec<OrderedField>,
    attributes: Vec<usize>,
}

impl TypeState {
    pub const fn new(
        results: Vec<OrderedField>,
        operands: Vec<OrderedField>,
        regions: Vec<OrderedField>,
        successors: Vec<OrderedField>,
        attributes: Vec<usize>,
    ) -> Self {
        Self {
            results,
            operands,
            regions,
            successors,
            attributes,
        }
    }

    fn ordered_fields(&self) -> impl Iterator<Item = (FieldKind, &[OrderedField])> {
        [
            (FieldKind::Result, self.results.as_slice()),
            (FieldKind::Operand, &self.operands),
            (FieldKind::Region, &self.regions),
            (FieldKind::Successor, &self.successors),
        ]
        .into_iter()
    }

    fn unordered_fields(&self) -> impl Iterator<Item = (FieldKind, &[usize])> {
        [(FieldKind::Attribute, self.attributes.as_slice())].into_iter()
    }

    pub fn has_parameter(&self, kind: FieldKind, index: usize) -> bool {
        self.ordered_fields()
            .any(|(other, fields)| other == kind && index < fields.len())
            || self
                .unordered_fields()
                .any(|(other, fields)| other == kind && fields.contains(&index))
    }

    pub fn parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .map(|(kind, fields)| (fields.len(), kind))
            .chain(
                self.unordered_fields()
                    .map(|(kind, fields)| (fields.len(), kind)),
            )
            .flat_map(|(count, kind)| Self::build_parameters(count, kind))
    }

    pub fn parameters_without(
        &self,
        kind: FieldKind,
        index: usize,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(move |(other, fields)| {
                Self::build_ordered_parameters_without(
                    fields,
                    other,
                    (other == kind).then_some(index),
                )
            })
            .chain(self.unordered_fields().flat_map(move |(other, fields)| {
                Self::build_unordered_parameters_without(
                    fields,
                    other,
                    (other == kind).then_some(index),
                )
            }))
    }

    pub fn arguments_with(
        &self,
        kind: FieldKind,
        index: usize,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(move |(other, fields)| {
                Self::build_ordered_arguments_with(
                    fields,
                    other,
                    (other == kind).then_some(index),
                    set,
                )
            })
            .chain(self.unordered_fields().flat_map(move |(other, fields)| {
                Self::build_unordered_arguments_with(
                    fields,
                    other,
                    (other == kind).then_some(index),
                    set,
                )
            }))
    }

    pub fn initial_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.parameters().map(|_| Self::build_argument(false))
    }

    pub fn final_parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(kind, fields)| Self::build_optional_parameters(fields, kind))
    }

    pub fn final_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(kind, fields)| Self::build_set_arguments(fields, kind))
            .chain(
                self.unordered_fields()
                    .flat_map(|(_, fields)| repeat_n(Self::build_argument(true), fields.len())),
            )
    }

    fn build_parameters(count: usize, kind: FieldKind) -> impl Iterator<Item = GenericArgument> {
        (0..count).map(move |index| Self::build_generic_argument(kind, index))
    }

    fn build_optional_parameters(
        fields: &[OrderedField],
        kind: FieldKind,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        fields
            .iter()
            .enumerate()
            .filter(|(_, field)| field.is_optional())
            .map(move |(index, _)| Self::build_generic_argument(kind, index))
    }

    fn build_set_arguments(
        fields: &[OrderedField],
        kind: FieldKind,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        fields.iter().enumerate().map(move |(index, field)| {
            if field.is_optional() {
                Self::build_generic_argument(kind, index)
            } else {
                Self::build_argument(true)
            }
        })
    }

    fn build_ordered_parameters_without(
        fields: &[OrderedField],
        kind: FieldKind,
        field: Option<usize>,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        if let Some(index) = field {
            Box::new(Self::build_optional_parameters(&fields[..index], kind))
                as Box<dyn Iterator<Item = _>>
        } else {
            Box::new(Self::build_parameters(fields.len(), kind))
        }
    }

    fn build_unordered_parameters_without(
        fields: &[usize],
        kind: FieldKind,
        field: Option<usize>,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        fields
            .iter()
            .enumerate()
            .filter(move |(_, other)| Some(**other) != field)
            .map(move |(index, _)| Self::build_generic_argument(kind, index))
    }

    fn build_ordered_arguments_with(
        fields: &[OrderedField],
        kind: FieldKind,
        field: Option<usize>,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        if let Some(index) = field {
            Box::new(
                Self::build_set_arguments(&fields[..index], kind)
                    .chain([Self::build_argument(set)])
                    .chain(repeat_n(
                        Self::build_argument(false),
                        fields.len() - index - 1,
                    )),
            ) as Box<dyn Iterator<Item = _>>
        } else {
            Box::new(Self::build_parameters(fields.len(), kind))
        }
    }

    fn build_unordered_arguments_with(
        fields: &[usize],
        kind: FieldKind,
        field: Option<usize>,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        fields.iter().enumerate().map(move |(index, other)| {
            if Some(*other) == field {
                Self::build_argument(set)
            } else {
                Self::build_generic_argument(kind, index)
            }
        })
    }

    fn build_generic_argument(kind: FieldKind, index: usize) -> GenericArgument {
        let identifier = format_ident!("{}{index}", kind.type_parameter_prefix());

        parse_quote!(#identifier)
    }

    fn build_argument(set: bool) -> GenericArgument {
        if set {
            parse_quote!(::melior::dialect::ods::__private::Set)
        } else {
            parse_quote!(::melior::dialect::ods::__private::Unset)
        }
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

    fn parameter(name: &str) -> GenericArgument {
        let identifier = format_ident!("{name}");

        parse_quote!(#identifier)
    }

    fn set() -> GenericArgument {
        parse_quote!(::melior::dialect::ods::__private::Set)
    }

    fn unset() -> GenericArgument {
        parse_quote!(::melior::dialect::ods::__private::Unset)
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
    fn build_parameters_without_ordered_field() {
        assert_eq!(
            create_type_state()
                .parameters_without(FieldKind::Result, 1)
                .collect::<Vec<_>>(),
            ["T0", "O0", "A0", "A1"].map(parameter)
        );
    }

    #[test]
    fn build_parameters_without_attribute() {
        assert_eq!(
            create_type_state()
                .parameters_without(FieldKind::Attribute, 2)
                .collect::<Vec<_>>(),
            ["T0", "T1", "O0", "A0"].map(parameter)
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
                set(),
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
                unset(),
                unset(),
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
                set()
            ]
        );
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
                set(),
                parameter("A0"),
                parameter("A1")
            ]
        );
    }
}
