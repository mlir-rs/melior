mod attribute;
mod builder;
mod element_position;
mod operand;
mod operation_element;
mod operation_field;
mod region;
mod result;
mod segment_kind;
mod successor;
mod variadic_kind;

pub use self::{
    attribute::Attribute, builder::OperationBuilder, element_position::ElementPosition,
    operand::Operand, operation_element::OperationElement, region::Region, result::OperationResult,
    segment_kind::SegmentKind, successor::Successor, variadic_kind::VariadicKind,
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
use tblgen::{TypedInit, error::WithLocation, record::Record};

// spell-checker: disable-next-line
const VOWELS: &str = "aeiou";

/// How a generated builder populates result types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeInference {
    /// Op implements `InferTypeOpInterface` — call
    /// `enable_result_type_inference()`.
    Interface,
    /// Op has `SameOperandsAndResultType` — copy `operands[0].type()` to all
    /// results in the first-operand setter.
    SameOperands,
    /// Op has `FirstAttrDerivedResultType` — derive result type from the first
    /// attribute in the first-attribute setter.
    FirstAttrDerived,
}

#[derive(Debug)]
pub struct Operation<'a> {
    name: String,
    short_name: String,
    short_dialect_name: &'a str,
    dialect_name: &'a str,
    operation_name: &'a str,
    constructor_identifier: Ident,
    summary: &'a str,
    description: String,
    type_inference: Option<TypeInference>,
    results: Vec<OperationResult>,
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
            &arguments,
            trait_names.contains("::mlir::OpTrait::SameVariadicResultSize"),
            trait_names.contains("::mlir::OpTrait::AttrSizedResultSegments"),
        )?;
        let short_name = Self::build_name(definition)?;

        Ok(Self {
            name: short_name.clone() + "Operation",
            short_name,
            dialect_name: definition.def_value("opDialect")?.name()?,
            short_dialect_name: definition.def_value("opDialect")?.str_value("name")?,
            operation_name,
            constructor_identifier: sanitize_snake_case_identifier(operation_name)?,
            summary: definition.str_value("summary")?,
            description: sanitize_documentation(definition.str_value("description")?)?,
            type_inference: {
                let has_interface = regions.is_empty()
                    && traits.iter().any(|r#trait| {
                        r#trait.name() == Some("::mlir::InferTypeOpInterface::Trait")
                    });
                let has_same_operands = unfixed_result_count == 0
                    && traits.iter().any(|r#trait| {
                        r#trait.name() == Some("::mlir::OpTrait::SameOperandsAndResultType")
                    });
                let has_first_attr_derived = unfixed_result_count == 0
                    && traits.iter().any(|r#trait| {
                        r#trait.name() == Some("::mlir::OpTrait::FirstAttrDerivedResultType")
                    });
                if has_interface {
                    Some(TypeInference::Interface)
                } else if has_same_operands {
                    Some(TypeInference::SameOperands)
                } else if has_first_attr_derived {
                    Some(TypeInference::FirstAttrDerived)
                } else {
                    None
                }
            },
            results,
            operands: Self::collect_operands(
                definition,
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

        let name = if let Some((_, name)) = name.split_once('_') {
            name
        } else {
            name
        };

        Ok(name.strip_suffix("Op").unwrap_or(name).to_owned())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn short_name(&self) -> &str {
        &self.short_name
    }

    pub const fn type_inference(&self) -> Option<TypeInference> {
        self.type_inference
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
        let article = VOWELS.contains(&self.operation_name()[..1]);

        format!(
            "{} [`{}`]({}) operation",
            if article { "an" } else { "a" },
            self.operation_name,
            self.name
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

    pub fn results(&self) -> impl Iterator<Item = &OperationResult> + Clone {
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

    // The segment size arrays a builder for this operation fills in, with their
    // lengths. Result types are not set by the builder if they are inferred, so
    // their group sizes cannot be computed either.
    pub fn segment_arrays(&self) -> impl Iterator<Item = (SegmentKind, usize)> {
        let operands = self
            .operands
            .iter()
            .any(|operand| operand.segment().is_some())
            .then_some((SegmentKind::Operand, self.operand_len()));
        let results = (self.type_inference.is_none()
            && self.results.iter().any(|result| result.segment().is_some()))
        .then_some((SegmentKind::Result, self.result_len()));

        operands.into_iter().chain(results)
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

    // Results whose types are given to builders rather than inferred.
    pub fn non_inferred_results(&self) -> impl Iterator<Item = &OperationResult> {
        if self.type_inference.is_some() {
            Default::default()
        } else {
            self.results.iter()
        }
    }

    pub fn required_results(&self) -> impl Iterator<Item = &OperationResult> {
        self.non_inferred_results()
            .filter(|field| !field.is_optional())
    }

    pub fn required_operands(&self) -> impl Iterator<Item = &'_ Operand<'_>> {
        self.operands.iter().filter(|field| !field.is_optional())
    }

    pub fn required_attributes(&self) -> impl Iterator<Item = &'_ Attribute<'_>> {
        self.attributes.iter().filter(|field| !field.is_optional())
    }

    pub fn required_fields(&self) -> impl Iterator<Item = &dyn OperationField> {
        fn convert(field: &impl OperationField) -> &dyn OperationField {
            field
        }

        self.required_results()
            .map(convert)
            .chain(self.required_operands().map(convert))
            .chain(self.regions().map(convert))
            .chain(self.successors().map(convert))
            .chain(self.required_attributes().map(convert))
    }

    fn collect_successors(definition: Record<'a>) -> Result<Vec<Successor<'a>>, Error> {
        definition
            .dag_value("successors")?
            .args()
            .map(|(name, value)| {
                let name = name.ok_or_else(|| {
                    OdsError::UnnamedDagArg("successors").with_location(definition)
                })?;
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
                let name = name
                    .ok_or_else(|| OdsError::UnnamedDagArg("regions").with_location(definition))?;
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
        let mut trait_lists = vec![definition.list_of_defs_value("traits")?];
        let mut traits = vec![];

        while let Some(trait_list) = trait_lists.pop() {
            for definition in trait_list {
                if definition.subclass_of("TraitList") {
                    trait_lists.push(definition.list_of_defs_value("traits")?);
                } else {
                    if definition.subclass_of("Interface") {
                        trait_lists.push(definition.list_of_defs_value("baseInterfaces")?);
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
    ) -> Result<Vec<(Option<&'a str>, Record<'a>)>, Error> {
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
        arguments: &[(Option<&'a str>, Record<'a>)],
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<OperationResult>, usize), Error> {
        let results = Self::dag_constraints(definition, "results")?;
        // Unnamed results (e.g. `(outs AnyType)` without `$name`) get generated
        // names so that they are preserved in builders and accessors.
        let taken = results
            .iter()
            .chain(arguments)
            .filter_map(|(name, _)| *name)
            .collect::<HashSet<_>>();
        let named = results
            .iter()
            .enumerate()
            .map(|(index, (name, constraint))| {
                let name = match name {
                    Some(name) => (*name).into(),
                    None => {
                        let name = format!("result_{index}");

                        if taken.contains(name.as_str()) {
                            return Err(OdsError::GeneratedNameConflict(name)
                                .with_location(definition)
                                .into());
                        }

                        name
                    }
                };

                Ok((name, Type::new(*constraint)))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Self::collect_elements(
            definition,
            &named
                .iter()
                .map(|(name, r#type)| (name.as_str(), *r#type))
                .collect::<Vec<_>>(),
            OperationResult::new,
            same_size,
            attribute_sized,
        )
    }

    fn collect_operands(
        definition: Record<'a>,
        arguments: &[(Option<&'a str>, Record<'a>)],
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<Vec<Operand<'a>>, Error> {
        Ok(Self::collect_elements(
            definition,
            &arguments
                .iter()
                .filter_map(|(name, definition)| Some(((*name)?, *definition)))
                .filter(|(_, definition)| definition.subclass_of("TypeConstraint"))
                .map(|(name, definition)| (name, Type::new(definition)))
                .collect::<Vec<_>>(),
            Operand::new,
            same_size,
            attribute_sized,
        )?
        .0)
    }

    fn collect_elements<'n, T>(
        definition: Record<'a>,
        elements: &[(&'n str, Type)],
        create: impl Fn(&'n str, Type, ElementPosition) -> Result<T, Error>,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<(Vec<T>, usize), Error> {
        let unfixed_count = Self::count_unfixed_elements(elements);
        let variadic_kind = if unfixed_count <= 1 {
            None
        } else if same_size {
            Some(VariadicKind::SameSize)
        } else if attribute_sized {
            Some(VariadicKind::AttributeSized)
        } else {
            // TODO: Support multiple variadic operands/results without these traits.
            return Err(OdsError::MissingVariadicSizeTrait
                .with_location(definition)
                .into());
        };

        Ok((
            elements
                .iter()
                .enumerate()
                .map(|(index, (name, r#type))| {
                    create(
                        name,
                        *r#type,
                        ElementPosition::new(
                            index,
                            Self::count_unfixed_elements(&elements[..index]),
                            unfixed_count,
                            variadic_kind,
                        ),
                    )
                })
                .collect::<Result<_, _>>()?,
            unfixed_count,
        ))
    }

    fn count_unfixed_elements(elements: &[(&str, Type)]) -> usize {
        elements
            .iter()
            .filter(|(_, r#type)| r#type.is_unfixed())
            .count()
    }

    fn collect_attributes(
        arguments: &[(Option<&'a str>, Record<'a>)],
    ) -> Result<Vec<Attribute<'a>>, Error> {
        arguments
            .iter()
            .filter_map(|(name, definition)| Some(((*name)?, *definition)))
            .filter(|(_, definition)| definition.subclass_of("Attr"))
            .map(|(name, definition)| {
                if definition.subclass_of("DerivedAttr") {
                    Err(OdsError::UnexpectedSuperClass("DerivedAttr")
                        .with_location(definition)
                        .into())
                } else {
                    Attribute::new(name, definition)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::LLVM_INCLUDE_DIRECTORY;
    use pretty_assertions::assert_eq;
    use tblgen::TableGenParser;

    #[test]
    fn fail_to_create_operation_without_variadic_size_trait() {
        let keeper = TableGenParser::new()
            .add_include_directory(LLVM_INCLUDE_DIRECTORY)
            .add_source(&unindent::unindent(
                r#"
                include "mlir/IR/OpBase.td"

                def Test_Dialect : Dialect {
                    let name = "test";
                }

                def Test_FooOp : Op<Test_Dialect, "foo"> {
                    let arguments = (ins Variadic<I32>:$lhs, Variadic<I32>:$rhs);
                }
                "#,
            ))
            .unwrap()
            .parse()
            .unwrap();

        assert_eq!(
            Operation::new(keeper.def("Test_FooOp").unwrap())
                .unwrap_err()
                .add_source_info(keeper.source_info())
                .to_string(),
            "invalid ODS input: error: multiple variadic operands/results require \
            SameVariadicOperandSize, SameVariadicResultSize, or \
            AttrSizedOperandSegments/AttrSizedResultSegments trait\n\
            def Test_FooOp : Op<Test_Dialect, \"foo\"> {\n    ^\n"
        );
    }

    #[test]
    fn preserve_unnamed_results() {
        let keeper = TableGenParser::new()
            .add_include_directory(LLVM_INCLUDE_DIRECTORY)
            .add_source(&unindent::unindent(
                r#"
                include "mlir/IR/OpBase.td"

                def Test_Dialect : Dialect {
                    let name = "test";
                }

                def Test_FooOp : Op<Test_Dialect, "foo"> {
                    let arguments = (ins I32:$input);
                    let results = (outs I32, I64);
                }
                "#,
            ))
            .unwrap()
            .parse()
            .unwrap();

        let operation = Operation::new(keeper.def("Test_FooOp").unwrap()).unwrap();

        assert_eq!(
            operation
                .results()
                .map(OperationField::name)
                .collect::<Vec<_>>(),
            ["result_0", "result_1"]
        );
    }

    #[test]
    fn fail_to_generate_conflicting_result_name() {
        let keeper = TableGenParser::new()
            .add_include_directory(LLVM_INCLUDE_DIRECTORY)
            .add_source(&unindent::unindent(
                r#"
                include "mlir/IR/OpBase.td"

                def Test_Dialect : Dialect {
                    let name = "test";
                }

                def Test_FooOp : Op<Test_Dialect, "foo"> {
                    let results = (outs I32:$result_1, I32);
                }
                "#,
            ))
            .unwrap()
            .parse()
            .unwrap();

        assert_eq!(
            Operation::new(keeper.def("Test_FooOp").unwrap())
                .unwrap_err()
                .add_source_info(keeper.source_info())
                .to_string(),
            "invalid ODS input: error: generated name result_1 for an unnamed \
            result conflicts with a named element\n\
            def Test_FooOp : Op<Test_Dialect, \"foo\"> {\n    ^\n"
        );
    }
}
