use crate::{
    context::Context,
    location::Location,
    operation::{Operation, OperationRef},
    r#type::Type,
    region::RegionRef,
    utility::into_raw_array,
    value::Value,
};
use mlir_sys::{
    mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockCreate, mlirBlockDestroy,
    mlirBlockEqual, mlirBlockGetArgument, mlirBlockGetFirstOperation, mlirBlockGetNumArguments,
    mlirBlockGetParentRegion, mlirBlockInsertOwnedOperation, MlirBlock,
};
use std::{marker::PhantomData, mem::forget, ops::Deref};

/// A block
#[derive(Debug)]
pub struct Block<'c> {
    r#ref: BlockRef<'static>,
    _context: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    /// Creates a block.
    pub fn new(arguments: &[(Type<'c>, Location<'c>)]) -> Self {
        unsafe {
            Self {
                r#ref: BlockRef::from_raw(mlirBlockCreate(
                    arguments.len() as isize,
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(argument, _)| argument.to_raw())
                            .collect(),
                    ),
                    into_raw_array(
                        arguments
                            .iter()
                            .map(|(_, location)| location.to_raw())
                            .collect(),
                    ),
                )),
                _context: Default::default(),
            }
        }
    }

    pub(crate) unsafe fn into_raw(self) -> MlirBlock {
        let block = self.raw;

        forget(self);

        block
    }
}

impl<'c> Drop for Block<'c> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.raw) };
    }
}

impl<'c> PartialEq for Block<'c> {
    fn eq(&self, other: &Self) -> bool {
        self.r#ref == other.r#ref
    }
}

impl<'c> Eq for Block<'c> {}

impl<'c> Deref for Block<'c> {
    type Target = BlockRef<'static>;

    fn deref(&self) -> &Self::Target {
        &self.r#ref
    }
}

/// A reference of a block.
// TODO Should we split context lifetimes? Or, is it transitively proven that 'c
// > 'a?
#[derive(Clone, Copy, Debug)]
pub struct BlockRef<'a> {
    raw: MlirBlock,
    _reference: PhantomData<&'a Block<'a>>,
}

impl<'c> BlockRef<'c> {
    /// Gets an argument at a position.
    pub fn argument(&self, position: usize) -> Option<Value> {
        unsafe {
            if position < self.argument_count() as usize {
                Some(Value::from_raw(mlirBlockGetArgument(
                    self.raw,
                    position as isize,
                )))
            } else {
                None
            }
        }
    }

    /// Gets a number of arguments.
    pub fn argument_count(&self) -> usize {
        unsafe { mlirBlockGetNumArguments(self.raw) as usize }
    }

    /// Gets a parent region.
    pub fn parent_region(&self) -> Option<RegionRef> {
        unsafe { RegionRef::from_option_raw(mlirBlockGetParentRegion(self.raw)) }
    }

    /// Gets the first operation.
    pub fn first_operation(&self) -> Option<OperationRef> {
        unsafe {
            let operation = mlirBlockGetFirstOperation(self.raw);

            if operation.ptr.is_null() {
                None
            } else {
                Some(OperationRef::from_raw(operation))
            }
        }
    }

    /// Adds an argument.
    pub fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.raw,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    /// Inserts an operation.
    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    pub fn insert_operation(&self, position: usize, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockInsertOwnedOperation(self.raw, position as isize, operation);

            OperationRef::from_raw(operation)
        }
    }

    /// Appends an operation.
    pub fn append_operation(&self, operation: Operation) -> OperationRef {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockAppendOwnedOperation(self.raw, operation);

            OperationRef::from_raw(operation)
        }
    }

    pub(crate) unsafe fn from_raw(raw: MlirBlock) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    pub(crate) unsafe fn from_option_raw(raw: MlirBlock) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    pub(crate) unsafe fn to_raw(self) -> MlirBlock {
        self.raw
    }
}

impl<'a> PartialEq for BlockRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirBlockEqual(self.raw, other.raw) }
    }
}

impl<'a> Eq for BlockRef<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{operation_state::OperationState, region::Region};

    #[test]
    fn new() {
        Block::new(&[]);
    }

    #[test]
    fn argument() {
        let context = Context::new();
        let r#type = Type::integer(&context, 64);

        assert_eq!(
            Block::new(&[(r#type, Location::unknown(&context))])
                .argument(0)
                .unwrap()
                .r#type(),
            r#type
        );
    }

    #[test]
    fn argument_none() {
        assert!(Block::new(&[]).argument(0).is_none());
    }

    #[test]
    fn argument_count() {
        assert_eq!(Block::new(&[]).argument_count(), 0);
    }

    #[test]
    fn parent_region() {
        let region = Region::new();
        let block = region.append_block(Block::new(&[]));

        assert_eq!(block.parent_region(), Some(*region));
    }

    #[test]
    fn parent_region_none() {
        let block = Block::new(&[]);

        assert_eq!(block.parent_region(), None);
    }

    #[test]
    fn first_operation() {
        let context = Context::new();
        let block = Block::new(&[]);

        let operation = block.append_operation(Operation::new(OperationState::new(
            "foo",
            Location::unknown(&context),
        )));

        assert_eq!(block.first_operation(), Some(operation));
    }

    #[test]
    fn first_operation_none() {
        let block = Block::new(&[]);

        assert_eq!(block.first_operation(), None);
    }

    #[test]
    fn append_operation() {
        let context = Context::new();
        let block = Block::new(&[]);

        block.append_operation(Operation::new(OperationState::new(
            "foo",
            Location::unknown(&context),
        )));
    }

    #[test]
    fn insert_operation() {
        let context = Context::new();
        let block = Block::new(&[]);

        block.insert_operation(
            0,
            Operation::new(OperationState::new("foo", Location::unknown(&context))),
        );
    }
}
