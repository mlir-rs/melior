mod attribute;
mod builder;
mod operand;
mod operation_element;
mod operation_field;
mod region;
mod result;
mod successor;
mod variadic_kind;

pub use self::{
    attribute::Attribute, builder::OperationBuilder, operand::Operand,
    operation_element::OperationElement, region::Region, result::OperationResult,
    successor::Successor, variadic_kind::VariadicKind,
};
use super::utility::{sanitize_documentation, sanitize_snake_case_identifier};
use crate::dialect::{
    error::{Error, OdsError},
    r#trait::Trait,
    r#type::Type,
    utility::capitalize_string,
};
pub use operation_field::OperationField;
use std::collections::HashSet;
use syn::Ident;
use tblgen::{error::WithLocation, record::Record, TypedInit};

// spell-checker: disable-next-line
const VOWELS: &str = "aeiou";

#[derive(Debug)]
pub struct Operation<'a> {
    name: String,
    short_dialect_name: &'a str,
    dialect_name: &'a str,
    operation_name: &'a str,
    constructor_identifier: Ident,
    summary: &'a str,
    description: String,
    can_infer_type: bool,
    results: Vec<OperationResult<'a>>,
    operands: Vec<Operand<'a>>,
    regions: Vec<Region<'a>>,
    successors: Vec<Successor<'a>>,
    attributes: Vec<Attribute<'a>>,
    derived_attributes: Vec<Attribute<'a>>,
}

impl<'a> Operation<'a> {
    pub fn new(definition: Record<'a>) -> Result<Self, Error> {
        let operation_name = definition.str_value("opName")?;
        let traits = Self::collect_traits(definition)?;
        let trait_names = traits
            .iter()
            .flat_map(|r#trait| r#trait.name())
            .collect::<HashSet<_>>();

        let arguments = Self::dag_constraints(definition, "arguments")?;
        let regions = Self::collect_regions(definition)?;
        let (results, unfixed_result_count) = Self::collect_results(
            definition,
            trait_names.contains("::mlir::OpTrait::SameVariadicResultSize"),
            trait_names.contains("::mlir::OpTrait::AttrSizedResultSegments"),
        )?;

        Ok(Self {
            name: Self::build_name(definition)?,
            dialect_name: definition.def_value("opDialect")?.name()?,
            short_dialect_name: definition.def_value("opDialect")?.str_value("name")?,
            operation_name,
            constructor_identifier: sanitize_snake_case_identifier(operation_name)?,
            summary: definition.str_value("summary")?,
            description: sanitize_documentation(definition.str_value("description")?)?,
            can_infer_type: traits.iter().any(|r#trait| {
                (r#trait.name() == Some("::mlir::OpTrait::FirstAttrDerivedResultType")
                    || r#trait.name() == Some("::mlir::OpTrait::SameOperandsAndResultType"))
                    && unfixed_result_count == 0
                    || r#trait.name() == Some("::mlir::InferTypeOpInterface::Trait")
                        && regions.is_empty()
            }),
            results,
            operands: Self::collect_operands(
                &arguments,
                trait_names.contains("::mlir::OpTrait::SameVariadicOperandSize"),
                trait_names.contains("::mlir::OpTrait::AttrSizedOperandSegments"),
            )?,
            regions,
            successors: Self::collect_successors(definition)?,
            attributes: Self::collect_attributes(&arguments)?,
            derived_attributes: Self::collect_derived_attributes(definition)?,
        })
    }

    fn build_name(definition: Record) -> Result<String, Error> {
        let name = definition.name()?;

        Ok(if let Some((_, name)) = name.split_once('_') {
            name
        } else {
            name
        }
        .trim_end_matches("Op")
        .to_owned()
            + "Operation")
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn can_infer_type(&self) -> bool {
        self.can_infer_type
    }

    pub const fn dialect_name(&self) -> &str {
        self.dialect_name
    }

    pub const fn operation_name(&self) -> &str {
        self.operation_name
    }

    pub fn full_operation_name(&self) -> String {
        format!("{}.{}", self.short_dialect_name, self.operation_name)
    }

    pub fn documentation_name(&self) -> String {
        format!(
            "{} [`{}`]({}) operation",
            if VOWELS.contains(&self.operation_name()[..1]) {
                "an"
            } else {
                "a"
            },
            self.operation_name,
            &self.name
        )
    }

    pub fn summary(&self) -> String {
        format!(
            "{}. {}",
            capitalize_string(&self.documentation_name()),
            if self.summary.is_empty() {
                Default::default()
            } else {
                capitalize_string(self.summary) + "."
            },
        )
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub const fn constructor_identifier(&self) -> &Ident {
        &self.constructor_identifier
    }

    pub fn results(&self) -> impl Iterator<Item = &OperationResult<'a>> + Clone {
        self.results.iter()
    }

    pub fn result_len(&self) -> usize {
        self.results.len()
    }

    pub fn operands(&self) -> impl Iterator<Item = &Operand<'a>> + Clone {
        self.operands.iter()
    }

    pub fn operand_len(&self) -> usize {
        self.operands.len()
    }

    pub fn regions(&self) -> impl Iterator<Item = &Region<'a>> {
        self.regions.iter()
    }

    pub fn successors(&self) -> impl Iterator<Item = &Successor<'a>> {
        self.successors.iter()
    }

    pub fn attributes(&self) -> impl Iterator<Item = &Attribute<'a>> {
        self.attributes.iter()
    }

    pub fn all_attributes(&self) -> impl Iterator<Item = &Attribute<'a>> {
        self.attributes().chain(&self.derived_attributes)
    }

    pub fn required_results(&self) -> impl Iterator<Item = &OperationResult<'_>> {
        if self.can_infer_type {
            Default::default()
        } else {
            self.results.iter()
        }
        .filter(|field| !field.is_optional())
    }

    pub fn required_operands(&self) -> impl Iterator<Item = &Operand<'_>> {
        self.operands.iter().filter(|field| !field.is_optional())
    }

    pub fn required_regions(&self) -> impl Iterator<Item = &Region<'_>> {
        self.regions.iter().filter(|field| !field.is_optional())
    }

    pub fn required_successors(&self) -> impl Iterator<Item = &Successor<'_>> {
        self.successors.iter().filter(|field| !field.is_optional())
    }

    pub fn required_attributes(&self) -> impl Iterator<Item = &Attribute<'_>> {
        self.attributes.iter().filter(|field| !field.is_optional())
    }

    pub fn required_fields(&self) -> impl Iterator<Item = &dyn OperationField> {
        fn convert(field: &impl OperationField) -> &dyn OperationField {
            field
        }

        self.required_results()
            .map(convert)
            .chain(self.required_operands().map(convert))
            .chain(self.required_regions().map(convert))
            .chain(self.required_successors().map(convert))
            .chain(self.required_attributes().map(convert))
    }

    fn collect_successors(definition: Record<'a>) -> Result<Vec<Successor<'a>>, Error> {
        definition
            .dag_value("successors")?
            .args()
            .map(|(name, value)| {
                Successor::new(
                    name,
                    Record::try_from(value)
                        .map_err(|error| error.set_location(definition))?
                        .subclass_of("VariadicSuccessor"),
                )
            })
            .collect()
    }

    fn collect_regions(definition: Record<'a>) -> Result<Vec<Region<'a>>, Error> {
        definition
            .dag_value("regions")?
            .args()
            .map(|(name, value)| {
                Region::new(
                    name,
                    Record::try_from(value)
                        .map_err(|error| error.set_location(definition))?
                        .subclass_of("VariadicRegion"),
                )
            })
            .collect()
    }

    fn collect_traits(definition: Record<'a>) -> Result<Vec<Trait>, Error> {
        let mut trait_lists = vec![definition.list_value("traits")?];
        let mut traits = vec![];

        while let Some(trait_list) = trait_lists.pop() {
            for value in trait_list.iter() {
                let definition =
                    Record::try_from(value).map_err(|error| error.set_location(definition))?;

                if definition.subclass_of("TraitList") {
                    trait_lists.push(definition.list_value("traits")?);
                } else {
                    if definition.subclass_of("Interface") {
                        trait_lists.push(definition.list_value("baseInterfaces")?);
                    }
                    traits.push(Trait::new(definition)?)
                }
            }
        }

        Ok(traits)
    }

    fn dag_constraints(
        definition: Record<'a>,
        name: &str,
    ) -> Result<Vec<(&'a str, Record<'a>)>, Error> {
        definition
            .dag_value(name)?
            .args()
            .map(|(name, argument)| {
                let definition =
                    Record::try_from(argument).map_err(|error| error.set_location(definition))?;

                Ok((
                    name,
                    if definition.subclass_of("OpVariable") {
                        definition.def_value("constraint")?
                    } else {
                        definition
                    },
                ))
            })
            .collect()
    }

    fn collect_results(
        definition: Record<'a>,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<OperationResult<'a>>, usize), Error> {
        Self::collect_elements(
            &Self::dag_constraints(definition, "results")?
                .into_iter()
                .map(|(name, constraint)| (name, Type::new(constraint)))
                .collect::<Vec<_>>(),
            OperationResult::new,
            same_size,
            attribute_sized,
        )
    }

    fn collect_operands(
        arguments: &[(&'a str, Record<'a>)],
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<Vec<Operand<'a>>, Error> {
        Ok(Self::collect_elements(
            &arguments
                .iter()
                .filter(|(_, definition)| definition.subclass_of("TypeConstraint"))
                .map(|(name, definition)| (*name, Type::new(*definition)))
                .collect::<Vec<_>>(),
            Operand::new,
            same_size,
            attribute_sized,
        )?
        .0)
    }

    fn collect_elements<T>(
        elements: &[(&'a str, Type)],
        create: impl Fn(&'a str, Type, VariadicKind) -> Result<T, Error>,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<T>, usize), Error> {
        let unfixed_count = elements
            .iter()
            .filter(|(_, r#type)| r#type.is_unfixed())
            .count();
        let mut variadic_kind = VariadicKind::new(unfixed_count, same_size, attribute_sized);
        let mut fields = vec![];

        for (name, r#type) in elements {
            fields.push(create(name, *r#type, variadic_kind.clone())?);

            match &mut variadic_kind {
                VariadicKind::Simple { unfixed_seen } => {
                    if r#type.is_unfixed() {
                        *unfixed_seen = true;
                    }
                }
                VariadicKind::SameSize {
                    preceding_simple_count,
                    preceding_variadic_count,
                    ..
                } => {
                    if r#type.is_unfixed() {
                        *preceding_variadic_count += 1;
                    } else {
                        *preceding_simple_count += 1;
                    }
                }
                VariadicKind::AttributeSized => {}
            }
        }

        Ok((fields, unfixed_count))
    }

    fn collect_attributes(
        arguments: &[(&'a str, Record<'a>)],
    ) -> Result<Vec<Attribute<'a>>, Error> {
        arguments
            .iter()
            .filter(|(_, definition)| definition.subclass_of("Attr"))
            .map(|(name, definition)| {
                if definition.subclass_of("DerivedAttr") {
                    Err(OdsError::UnexpectedSuperClass("DerivedAttr")
                        .with_location(*definition)
                        .into())
                } else {
                    Attribute::new(name, *definition)
                }
            })
            .collect()
    }

    fn collect_derived_attributes(definition: Record<'a>) -> Result<Vec<Attribute<'a>>, Error> {
        definition
            .values()
            .filter(|value| matches!(value.init, TypedInit::Def(_)))
            .map(Record::try_from)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|definition| definition.subclass_of("Attr"))
            .map(|definition| {
                if definition.subclass_of("DerivedAttr") {
                    Attribute::new(definition.name()?, definition)
                } else {
                    Err(OdsError::ExpectedSuperClass("DerivedAttr")
                        .with_location(definition)
                        .into())
                }
            })
            .collect()
    }
}
