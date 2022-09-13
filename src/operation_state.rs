use crate::{
    attribute::Attribute, block::Block, context::Context, identifier::Identifier,
    location::Location, r#type::Type, region::Region, string_ref::StringRef,
    utility::into_raw_array, value::Value,
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateGet, MlirOperationState,
};
use std::marker::PhantomData;

pub struct OperationState<'c> {
    state: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> OperationState<'c> {
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            state: unsafe {
                mlirOperationStateGet(StringRef::from(name).to_raw(), location.to_raw())
            },
            _context: Default::default(),
        }
    }

    pub fn add_results(mut self, results: Vec<Type<'c>>) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.state,
                results.len() as isize,
                into_raw_array(results.iter().map(|r#type| r#type.to_raw()).collect()),
            )
        }

        self
    }

    pub fn add_operands(mut self, operands: Vec<Value>) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.state,
                operands.len() as isize,
                into_raw_array(operands.iter().map(|value| value.to_raw()).collect()),
            )
        }

        self
    }

    pub fn add_regions(mut self, regions: Vec<Region>) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.state,
                regions.len() as isize,
                into_raw_array(
                    regions
                        .into_iter()
                        .map(|region| region.into_raw())
                        .collect(),
                ),
            )
        }

        self
    }

    pub fn add_successors(mut self, successors: Vec<Block>) -> Self {
        unsafe {
            mlirOperationStateAddSuccessors(
                &mut self.state,
                successors.len() as isize,
                into_raw_array(
                    successors
                        .into_iter()
                        .map(|block| block.into_raw())
                        .collect(),
                ),
            )
        }

        self
    }

    pub fn add_attributes(mut self, attributes: Vec<(Identifier, Attribute<'c>)>) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.state,
                attributes.len() as isize,
                into_raw_array(
                    attributes
                        .into_iter()
                        .map(|(identifier, attribute)| {
                            mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw())
                        })
                        .collect(),
                ),
            )
        }

        self
    }

    pub(crate) unsafe fn into_raw(self) -> MlirOperationState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, operation::Operation};

    #[test]
    fn new() {
        Operation::new(OperationState::new(
            "foo",
            Location::unknown(&Context::new()),
        ));
    }

    #[test]
    fn add_results() {
        let context = Context::new();

        Operation::new(
            OperationState::new("foo", Location::unknown(&context))
                .add_results(vec![Type::parse(&context, "i1")]),
        );
    }

    #[test]
    fn add_regions() {
        let context = Context::new();

        Operation::new(
            OperationState::new("foo", Location::unknown(&context))
                .add_regions(vec![Region::new()]),
        );
    }

    #[test]
    fn add_successors() {
        let context = Context::new();

        Operation::new(
            OperationState::new("foo", Location::unknown(&context))
                .add_successors(vec![Block::new(vec![])]),
        );
    }

    #[test]
    fn add_attributes() {
        let context = Context::new();

        Operation::new(
            OperationState::new("foo", Location::unknown(&context)).add_attributes(vec![(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit"),
            )]),
        );
    }
}
