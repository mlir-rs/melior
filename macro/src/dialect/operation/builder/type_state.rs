use quote::format_ident;
use std::iter::repeat_n;
use syn::{GenericArgument, parse_quote};

const RESULT_PREFIX: &str = "T";
const OPERAND_PREFIX: &str = "O";
const REGION_PREFIX: &str = "R";
const SUCCESSOR_PREFIX: &str = "S";
const ATTRIBUTE_PREFIX: &str = "A";

// A field whose setter appends elements to an operation. Its setter must be
// called before the ones of the later fields of the same kind.
#[derive(Debug)]
pub struct OrderedField {
    name: String,
    optional: bool,
}

impl OrderedField {
    pub const fn new(name: String, optional: bool) -> Self {
        Self { name, optional }
    }
}

#[derive(Debug)]
pub struct TypeState {
    results: Vec<OrderedField>,
    operands: Vec<OrderedField>,
    regions: Vec<OrderedField>,
    successors: Vec<OrderedField>,
    attributes: Vec<String>,
}

impl TypeState {
    pub const fn new(
        results: Vec<OrderedField>,
        operands: Vec<OrderedField>,
        regions: Vec<OrderedField>,
        successors: Vec<OrderedField>,
        attributes: Vec<String>,
    ) -> Self {
        Self {
            results,
            operands,
            regions,
            successors,
            attributes,
        }
    }

    fn ordered_fields(&self) -> impl Iterator<Item = (&[OrderedField], &'static str)> {
        [
            (self.results.as_slice(), RESULT_PREFIX),
            (&self.operands, OPERAND_PREFIX),
            (&self.regions, REGION_PREFIX),
            (&self.successors, SUCCESSOR_PREFIX),
        ]
        .into_iter()
    }

    fn unordered_fields(&self) -> impl Iterator<Item = (&[String], &'static str)> {
        [(self.attributes.as_slice(), ATTRIBUTE_PREFIX)].into_iter()
    }

    pub fn has_parameter(&self, field: &str) -> bool {
        self.ordered_fields()
            .any(|(fields, _)| fields.iter().any(|other| other.name == field))
            || self
                .unordered_fields()
                .any(|(fields, _)| fields.iter().any(|other| other == field))
    }

    pub fn parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .map(|(fields, prefix)| (fields.len(), prefix))
            .chain(
                self.unordered_fields()
                    .map(|(fields, prefix)| (fields.len(), prefix)),
            )
            .flat_map(|(count, prefix)| Self::build_parameters(count, prefix))
    }

    pub fn parameters_without<'a>(
        &'a self,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        self.ordered_fields()
            .flat_map(|(fields, prefix)| {
                Self::build_ordered_parameters_without(fields, prefix, field)
            })
            .chain(self.unordered_fields().flat_map(|(fields, prefix)| {
                Self::build_unordered_parameters_without(fields, prefix, field)
            }))
    }

    pub fn arguments_with<'a>(
        &'a self,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        self.ordered_fields()
            .flat_map(move |(fields, prefix)| {
                Self::build_ordered_arguments_with(fields, prefix, field, set)
            })
            .chain(self.unordered_fields().flat_map(move |(fields, prefix)| {
                Self::build_unordered_arguments_with(fields, prefix, field, set)
            }))
    }

    // The arguments of a new builder, where no field is set.
    pub fn initial_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.parameters().map(|_| Self::build_argument(false))
    }

    // The parameters and arguments of a builder ready to build an operation,
    // where all the required fields are set.
    pub fn final_parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(fields, prefix)| Self::build_optional_parameters(fields, prefix))
    }

    pub fn final_arguments(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.ordered_fields()
            .flat_map(|(fields, prefix)| Self::build_set_arguments(fields, prefix))
            .chain(
                self.unordered_fields()
                    .flat_map(|(fields, _)| repeat_n(Self::build_argument(true), fields.len())),
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
            .filter(|(_, field)| field.optional)
            .map(|(index, _)| Self::build_generic_argument(prefix, index))
    }

    // Required fields are set while optional ones are either set or unset.
    fn build_set_arguments<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields.iter().enumerate().map(|(index, field)| {
            if field.optional {
                Self::build_generic_argument(prefix, index)
            } else {
                Self::build_argument(true)
            }
        })
    }

    fn build_ordered_parameters_without<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
        field: &'a str,
    ) -> Box<dyn Iterator<Item = GenericArgument> + 'a> {
        let Some(index) = fields.iter().position(|other| other.name == field) else {
            return Box::new(Self::build_parameters(fields.len(), prefix));
        };

        Box::new(Self::build_optional_parameters(&fields[..index], prefix))
    }

    fn build_unordered_parameters_without<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields
            .iter()
            .enumerate()
            .filter(move |(_, other)| *other != field)
            .map(|(index, _)| Self::build_generic_argument(prefix, index))
    }

    // The later fields are unset so that a setter cannot be called after their
    // setters.
    fn build_ordered_arguments_with<'a>(
        fields: &'a [OrderedField],
        prefix: &'a str,
        field: &'a str,
        set: bool,
    ) -> Box<dyn Iterator<Item = GenericArgument> + 'a> {
        let Some(index) = fields.iter().position(|other| other.name == field) else {
            return Box::new(Self::build_parameters(fields.len(), prefix));
        };

        Box::new(
            Self::build_set_arguments(&fields[..index], prefix)
                .chain([Self::build_argument(set)])
                .chain(repeat_n(
                    Self::build_argument(false),
                    fields.len() - index - 1,
                )),
        )
    }

    fn build_unordered_arguments_with<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        fields.iter().enumerate().map(move |(index, other)| {
            if other == field {
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
