use super::Operation;
use crate::{
    Error,
    ir::{
        Attribute, AttributeLike, Block, Identifier, Location, Region, Type, TypeLike, Value,
        ValueLike,
    },
};
use mlir_sys::{
    MlirBlock, MlirNamedAttribute, MlirOperationState, MlirType, MlirValue, mlirNamedAttributeGet,
    mlirOperationCreate, mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet,
};

/// An operation builder.
pub struct OperationBuilder<'c> {
    name: Identifier<'c>,
    location: Location<'c>,
    results: Vec<MlirType>,
    operands: Vec<MlirValue>,
    regions: Vec<Region<'c>>,
    successors: Vec<MlirBlock>,
    attributes: Vec<MlirNamedAttribute>,
    result_type_inference: bool,
}

impl<'c> OperationBuilder<'c> {
    /// Creates an operation builder.
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            name: Identifier::new(unsafe { location.context().to_ref() }, name),
            location,
            results: vec![],
            operands: vec![],
            regions: vec![],
            successors: vec![],
            attributes: vec![],
            result_type_inference: false,
        }
    }

    /// Adds results.
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        self.results
            .extend(results.iter().map(|r#type| r#type.to_raw()));

        self
    }

    /// Adds operands.
    pub fn add_operands(mut self, operands: &[Value<'c, '_>]) -> Self {
        self.operands
            .extend(operands.iter().map(|operand| operand.to_raw()));

        self
    }

    /// Adds regions.
    pub fn add_regions<const N: usize>(mut self, regions: [Region<'c>; N]) -> Self {
        self.regions.extend(regions);

        self
    }

    /// Adds regions in a [`Vec`](std::vec::Vec).
    pub fn add_regions_vec(mut self, regions: Vec<Region<'c>>) -> Self {
        self.regions.extend(regions);

        self
    }

    /// Adds successor blocks.
    ///
    /// # Safety
    ///
    /// The successor blocks must not be destroyed while the operation refers to
    /// them.
    // TODO Provide a safe way to add successors, such as inserting operations
    // into blocks in the same regions as their successors.
    pub unsafe fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
        self.successors
            .extend(successors.iter().map(|block| block.to_raw()));

        self
    }

    /// Adds attributes.
    pub fn add_attributes(mut self, attributes: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
        self.attributes
            .extend(attributes.iter().map(|(identifier, attribute)| unsafe {
                mlirNamedAttributeGet(identifier.to_raw(), attribute.to_raw())
            }));

        self
    }

    /// Enables result type inference.
    pub const fn enable_result_type_inference(mut self) -> Self {
        self.result_type_inference = true;

        self
    }

    /// Builds an operation.
    pub fn build(self) -> Result<Operation<'c>, Error> {
        unsafe {
            let mut state =
                mlirOperationStateGet(self.name.as_string_ref().to_raw(), self.location.to_raw());

            add_elements(&mut state, mlirOperationStateAddResults, &self.results);
            add_elements(&mut state, mlirOperationStateAddOperands, &self.operands);
            add_elements(
                &mut state,
                mlirOperationStateAddOwnedRegions,
                &self
                    .regions
                    .into_iter()
                    .map(Region::into_raw)
                    .collect::<Vec<_>>(),
            );
            add_elements(
                &mut state,
                mlirOperationStateAddSuccessors,
                &self.successors,
            );
            add_elements(
                &mut state,
                mlirOperationStateAddAttributes,
                &self.attributes,
            );

            if self.result_type_inference {
                mlirOperationStateEnableResultTypeInference(&mut state);
            }

            Operation::from_option_raw(mlirOperationCreate(&mut state))
        }
        .ok_or(Error::OperationBuild)
    }
}

unsafe fn add_elements<T>(
    state: &mut MlirOperationState,
    add: unsafe extern "C" fn(*mut MlirOperationState, isize, *const T),
    elements: &[T],
) {
    if !elements.is_empty() {
        unsafe { add(state, elements.len() as isize, elements.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Block, RegionLike, ValueLike, block::BlockLike, operation::OperationLike},
        test::create_test_context,
    };
    use mlir_sys::{mlirOpOperandIsNull, mlirValueGetFirstUse};

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();
    }

    #[test]
    fn new_with_dropped_name() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let builder = OperationBuilder::new(&String::from("foo.bar"), Location::unknown(&context));
        let _name = String::from("baz.qux");

        assert_eq!(
            builder.build().unwrap().name(),
            Identifier::new(&context, "foo.bar")
        );
    }

    #[test]
    fn add_operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[r#type], &[location]).unwrap();
        let argument = block.argument(0).unwrap().into();

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_operands(&[argument])
            .build()
            .unwrap();
    }

    #[test]
    fn add_results() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[Type::parse(&context, "i1").unwrap()])
            .build()
            .unwrap();
    }

    #[test]
    fn add_regions() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions([Region::new()])
            .build()
            .unwrap();
    }

    #[test]
    fn drop_regions_without_build() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let block = Block::new(&[Type::index(&context)], &[location]).unwrap();
        let region = Region::new();
        let inner_block = Block::new(&[], &[]).unwrap();

        inner_block.append_operation(
            OperationBuilder::new("foo", location)
                .add_operands(&[block.argument(0).unwrap().into()])
                .build()
                .unwrap(),
        );
        region.append_block(inner_block);

        drop(OperationBuilder::new("bar", location).add_regions([region]));

        assert!(unsafe {
            mlirOpOperandIsNull(mlirValueGetFirstUse(block.argument(0).unwrap().to_raw()))
        });
    }

    #[test]
    fn add_successors() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[], &[]).unwrap();

        unsafe {
            OperationBuilder::new("foo", Location::unknown(&context)).add_successors(&[&block])
        }
        .build()
        .unwrap();
    }

    #[test]
    fn add_attributes() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit").unwrap(),
            )])
            .build()
            .unwrap();
    }

    #[test]
    fn enable_result_type_inference() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[r#type], &[location]).unwrap();
        let argument = block.argument(0).unwrap().into();

        assert_eq!(
            OperationBuilder::new("arith.addi", location)
                .add_operands(&[argument, argument])
                .enable_result_type_inference()
                .build()
                .unwrap()
                .result(0)
                .unwrap()
                .r#type(),
            r#type,
        );
    }
}
