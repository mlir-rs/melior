mod ordered_field;

pub use self::ordered_field::OrderedField;
use crate::dialect::operation::FieldKind;
use core::iter::repeat_n;
use quote::format_ident;
use syn::{GenericArgument, parse_quote};

const RESULT_PREFIX: &str = "T";
const OPERAND_PREFIX: &str = "O";
const REGION_PREFIX: &str = "R";
const SUCCESSOR_PREFIX: &str = "S";
const ATTRIBUTE_PREFIX: &str = "A";

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

    fn ordered_fields(&self) -> impl Iterator<Item = (FieldKind, &[OrderedField], &'static str)> {
        [
            (FieldKind::Result, self.results.as_slice(), RESULT_PREFIX),
            (FieldKind::Operand, &self.operands, OPERAND_PREFIX),
            (FieldKind::Region, &self.regions, REGION_PREFIX),
            (FieldKind::Successor, &self.successors, SUCCESSOR_PREFIX),
        ]
        .into_iter()
    }

    fn unordered_fields(&self) -> impl Iterator<Item = (FieldKind, &[usize], &'static str)> {
        [(
            FieldKind::Attribute,
            self.attributes.as_slice(),
            ATTRIBUTE_PREFIX,
        )]
        .into_iter()
    }

    pub fn has_parameter(&self, kind: FieldKind, index: usize) -> bool {
        self.ordered_fields()
            .any(|(other, fields, _)| other == kind && index < fields.len())
            || self
                .unordered_fields()
                .any(|(other, fields, _)| other == kind && fields.contains(&index))
    }

    pub fn parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .map(|(_, fields, prefix)| (fields.len(), prefix))
            .chain(
                self.unordered_fields()
                    .map(|(_, fields, prefix)| (fields.len(), prefix)),
            )
            .flat_map(|(count, prefix)| Self::build_parameters(count, prefix))
    }

    pub fn parameters_without(
        &self,
        kind: FieldKind,
        index: usize,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(move |(other, fields, prefix)| {
                Self::build_ordered_parameters_without(
                    fields,
                    prefix,
                    (other == kind).then_some(index),
                )
            })
            .chain(
                self.unordered_fields()
                    .flat_map(move |(other, fields, prefix)| {
                        Self::build_unordered_parameters_without(
                            fields,
                            prefix,
                            (other == kind).then_some(index),
                        )
                    }),
            )
    }

    pub fn arguments_with(
        &self,
        kind: FieldKind,
        index: usize,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(move |(other, fields, prefix)| {
                Self::build_ordered_arguments_with(
                    fields,
                    prefix,
                    (other == kind).then_some(index),
                    set,
                )
            })
            .chain(
                self.unordered_fields()
                    .flat_map(move |(other, fields, prefix)| {
                        Self::build_unordered_arguments_with(
                            fields,
                            prefix,
                            (other == kind).then_some(index),
                            set,
                        )
                    }),
            )
    }

    pub fn initial_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.parameters().map(|_| Self::build_argument(false))
    }

    pub fn final_parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(_, fields, prefix)| Self::build_optional_parameters(fields, prefix))
    }

    pub fn final_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(_, fields, prefix)| Self::build_set_arguments(fields, prefix))
            .chain(
                self.unordered_fields()
                    .flat_map(|(_, fields, _)| repeat_n(Self::build_argument(true), fields.len())),
            )
    }

    fn build_parameters(count: usize, prefix: &str) -> impl Iterator<Item = GenericArgument> + '_ {
        (0..count).map(|index| Self::build_generic_argument(prefix, index))
    }

    fn build_optional_parameters<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields
            .iter()
            .enumerate()
            .filter(|(_, field)| field.is_optional())
            .map(|(index, _)| Self::build_generic_argument(prefix, index))
    }

    fn build_set_arguments<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields.iter().enumerate().map(|(index, field)| {
            if field.is_optional() {
                Self::build_generic_argument(prefix, index)
            } else {
                Self::build_argument(true)
            }
        })
    }

    fn build_ordered_parameters_without<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
        field: Option<usize>,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        if let Some(index) = field {
            Box::new(Self::build_optional_parameters(&fields[..index], prefix))
                as Box<dyn Iterator<Item = _>>
        } else {
            Box::new(Self::build_parameters(fields.len(), prefix))
        }
    }

    fn build_unordered_parameters_without<'a>(
        fields: &'a [usize],
        prefix: &'a str,
        field: Option<usize>,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields
            .iter()
            .enumerate()
            .filter(move |(_, other)| Some(**other) != field)
            .map(|(index, _)| Self::build_generic_argument(prefix, index))
    }

    fn build_ordered_arguments_with<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
        field: Option<usize>,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        if let Some(index) = field {
            Box::new(
                Self::build_set_arguments(&fields[..index], prefix)
                    .chain([Self::build_argument(set)])
                    .chain(repeat_n(
                        Self::build_argument(false),
                        fields.len() - index - 1,
                    )),
            ) as Box<dyn Iterator<Item = _>>
        } else {
            Box::new(Self::build_parameters(fields.len(), prefix))
        }
    }

    fn build_unordered_arguments_with<'a>(
        fields: &'a [usize],
        prefix: &'a str,
        field: Option<usize>,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields.iter().enumerate().map(move |(index, other)| {
            if Some(*other) == field {
                Self::build_argument(set)
            } else {
                Self::build_generic_argument(prefix, index)
            }
        })
    }

    fn build_generic_argument(prefix: &str, index: usize) -> GenericArgument {
        let identifier = format_ident!("{prefix}{index}");

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
