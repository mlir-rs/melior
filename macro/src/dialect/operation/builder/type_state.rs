use quote::format_ident;
use std::iter::repeat_n;
use syn::{parse_quote, GenericArgument};

const RESULT_PREFIX: &str = "T";
const OPERAND_PREFIX: &str = "O";
const REGION_PREFIX: &str = "R";
const SUCCESSOR_PREFIX: &str = "S";
const ATTRIBUTE_PREFIX: &str = "A";

#[derive(Debug)]
pub struct TypeState {
    results: Vec<String>,
    operands: Vec<String>,
    regions: Vec<String>,
    successors: Vec<String>,
    attributes: Vec<String>,
}

impl TypeState {
    pub const fn new(
        results: Vec<String>,
        operands: Vec<String>,
        regions: Vec<String>,
        successors: Vec<String>,
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

    fn ordered_fields(&self) -> impl Iterator<Item = (&[String], &'static str)> {
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

    fn all_fields(&self) -> impl Iterator<Item = (&[String], &'static str)> {
        self.ordered_fields().chain(self.unordered_fields())
    }

    pub fn parameters(&self) -> impl Iterator<Item = GenericArgument> + '_ {
        self.all_fields()
            .flat_map(|(fields, prefix)| Self::build_parameters(fields, prefix))
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

    pub fn arguments_with_all(&self, set: bool) -> impl Iterator<Item = GenericArgument> + '_ {
        self.all_fields()
            .flat_map(move |(fields, _)| Self::build_arguments_with_all(fields, set))
    }

    fn build_parameters<'a>(
        fields: &[String],
        prefix: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        (0..fields.len()).map(|index| Self::build_generic_argument(prefix, index))
    }

    fn build_ordered_parameters_without<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
    ) -> impl Iterator<Item = GenericArgument> + 'a {
        Self::build_parameters(fields, prefix).skip(
            fields
                .iter()
                .position(|other| *other == field)
                .map(|index| index + 1)
                .unwrap_or(0),
        )
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

    fn build_ordered_arguments_with<'a>(
        fields: &'a [String],
        prefix: &'a str,
        field: &'a str,
        set: bool,
    ) -> Box<dyn Iterator<Item = GenericArgument> + 'a> {
        let Some(index) = fields.iter().position(|other| *other == field) else {
            return Box::new(Self::build_parameters(fields, prefix));
        };

        Box::new(
            repeat_n(Self::build_argument(true), index)
                .chain([Self::build_argument(set)])
                .chain(Self::build_parameters(fields, prefix).skip(index + 1)),
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

    fn build_arguments_with_all(
        fields: &[String],
        set: bool,
    ) -> impl Iterator<Item = GenericArgument> {
        repeat_n(Self::build_argument(set), fields.len())
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
