use super::Operation;
use crate::{
    Error,
    context::Context,
    ir::{Attribute, AttributeLike, Block, Identifier, Location, Region, Type, Value},
    string_ref::StringRef,
};
use mlir_sys::{
    MlirOperationState, mlirNamedAttributeGet, mlirOperationCreate,
    mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet,
};
use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, forget, transmute},
};

/// An operation builder.
pub struct OperationBuilder<'c> {
    raw: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> OperationBuilder<'c> {
    /// Creates an operation builder.
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            raw: unsafe { mlirOperationStateGet(StringRef::new(name).to_raw(), location.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Adds results.
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                results.as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds operands.
    pub fn add_operands(mut self, operands: &[Value<'c, '_>]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.raw,
                operands.len() as isize,
                operands.as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds regions.
    pub fn add_regions<const N: usize>(mut self, regions: [Region<'c>; N]) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                regions.as_ptr() as *const _,
            )
        }

        forget(regions);

        self
    }

    /// Adds regions in a [`Vec`](std::vec::Vec).
    pub fn add_regions_vec(mut self, regions: Vec<Region<'c>>) -> Self {
        unsafe {
            // This may fire with -D clippy::nursery, however, it is
            // guaranteed by the std that ManuallyDrop<T> has the same layout as T
            #[allow(clippy::transmute_undefined_repr)]
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                transmute::<Vec<Region>, Vec<ManuallyDrop<Region>>>(regions).as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds successor blocks.
    // TODO Fix this to ensure blocks are alive while they are referenced by the
    // operation.
    pub fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
        for block in successors {
            unsafe {
                mlirOperationStateAddSuccessors(&mut self.raw, 1, &[block.to_raw()] as *const _)
            }
        }

        self
    }

    /// Adds attributes.
    pub fn add_attributes(mut self, attributes: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
        for (identifier, attribute) in attributes {
            unsafe {
                mlirOperationStateAddAttributes(
                    &mut self.raw,
                    1,
                    &[mlirNamedAttributeGet(
                        identifier.to_raw(),
                        attribute.to_raw(),
                    )] as *const _,
                )
            }
        }

        self
    }

    /// Enables result type inference.
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.raw) }

        self
    }

    /// Builds an operation.
    pub fn build(mut self) -> Result<Operation<'c>, Error> {
        unsafe { Operation::from_option_raw(mlirOperationCreate(&mut self.raw)) }
            .ok_or(Error::OperationBuild)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Block, ValueLike, block::BlockLike, operation::OperationLike},
        test::create_test_context,
    };

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();
    }

    #[test]
    fn add_operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
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
    fn add_successors() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_successors(&[&Block::new(&[])])
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
        let block = Block::new(&[(r#type, location)]);
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
