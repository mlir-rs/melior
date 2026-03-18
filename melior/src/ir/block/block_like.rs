use super::{BlockArgument, BlockRef, TypeLike};
use crate::{
    Error,
    ir::{Location, Operation, OperationRef, RegionRef, Type, Value, operation::OperationRefMut},
};
use core::fmt::Display;
use mlir_sys::{
    MlirBlock, mlirBlockAddArgument, mlirBlockAppendOwnedOperation, mlirBlockEraseArgument,
    mlirBlockGetArgument, mlirBlockGetFirstOperation, mlirBlockGetNextInRegion,
    mlirBlockGetNumArguments, mlirBlockGetNumPredecessors, mlirBlockGetNumSuccessors,
    mlirBlockGetParentOperation, mlirBlockGetParentRegion, mlirBlockGetPredecessor,
    mlirBlockGetSuccessor, mlirBlockGetTerminator, mlirBlockInsertArgument,
    mlirBlockInsertOwnedOperation, mlirBlockInsertOwnedOperationAfter,
    mlirBlockInsertOwnedOperationBefore,
};

/// A trait for block-like types.
// TODO Split this trait into `BlockLike` and `BlockLikeMut`.
pub trait BlockLike<'c, 'a>: Display + 'a {
    /// Converts a block into a raw object.
    fn to_raw(&self) -> MlirBlock;

    /// Returns an argument at a position.
    fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'a>, Error> {
        unsafe {
            if index < self.argument_count() {
                Ok(BlockArgument::from_raw(mlirBlockGetArgument(
                    self.to_raw(),
                    index as isize,
                )))
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "block argument",
                    value: self.to_string(),
                    index,
                })
            }
        }
    }

    /// Returns a number of arguments.
    fn argument_count(&self) -> usize {
        unsafe { mlirBlockGetNumArguments(self.to_raw()) as usize }
    }

    /// Returns a reference to the first operation.
    fn first_operation(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetFirstOperation(self.to_raw())) }
    }

    /// Returns a mutable reference to the first operation.
    fn first_operation_mut(&self) -> Option<OperationRefMut<'c, 'a>> {
        unsafe { OperationRefMut::from_option_raw(mlirBlockGetFirstOperation(self.to_raw())) }
    }

    /// Returns a reference to a terminator operation.
    fn terminator(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetTerminator(self.to_raw())) }
    }

    /// Returns a mutable reference to a terminator operation.
    fn terminator_mut(&self) -> Option<OperationRefMut<'c, 'a>> {
        unsafe { OperationRefMut::from_option_raw(mlirBlockGetTerminator(self.to_raw())) }
    }

    /// Returns a parent region.
    // TODO Store lifetime of regions in blocks, or create another type like
    // `InsertedBlockRef`?
    fn parent_region(&self) -> Option<RegionRef<'c, 'a>> {
        unsafe { RegionRef::from_option_raw(mlirBlockGetParentRegion(self.to_raw())) }
    }

    /// Returns a parent operation.
    fn parent_operation(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirBlockGetParentOperation(self.to_raw())) }
    }

    /// Adds an argument.
    fn add_argument(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'a> {
        unsafe {
            Value::from_raw(mlirBlockAddArgument(
                self.to_raw(),
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    /// Appends an operation.
    fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'a> {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockAppendOwnedOperation(self.to_raw(), operation);

            OperationRef::from_raw(operation)
        }
    }

    /// Inserts an operation.
    // TODO How can we make those update functions take `&mut self`?
    // TODO Use cells?
    fn insert_operation(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'a> {
        unsafe {
            let operation = operation.into_raw();

            mlirBlockInsertOwnedOperation(self.to_raw(), position as isize, operation);

            OperationRef::from_raw(operation)
        }
    }

    /// Inserts an operation after another.
    fn insert_operation_after(
        &self,
        one: OperationRef<'c, 'a>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'a> {
        unsafe {
            let other = other.into_raw();

            mlirBlockInsertOwnedOperationAfter(self.to_raw(), one.to_raw(), other);

            OperationRef::from_raw(other)
        }
    }

    /// Inserts an operation before another.
    fn insert_operation_before(
        &self,
        one: OperationRef<'c, 'a>,
        other: Operation<'c>,
    ) -> OperationRef<'c, 'a> {
        unsafe {
            let other = other.into_raw();

            mlirBlockInsertOwnedOperationBefore(self.to_raw(), one.to_raw(), other);

            OperationRef::from_raw(other)
        }
    }

    /// Inserts an argument at the given position.
    ///
    /// Existing arguments at and after `index` are shifted. Any previously
    /// obtained `BlockArgument` handles for those positions become stale.
    fn insert_argument(
        &self,
        index: usize,
        r#type: Type<'c>,
        location: Location<'c>,
    ) -> Value<'c, 'a> {
        unsafe {
            Value::from_raw(mlirBlockInsertArgument(
                self.to_raw(),
                index as isize,
                r#type.to_raw(),
                location.to_raw(),
            ))
        }
    }

    /// Erases the argument at the given position.
    ///
    /// # Safety
    ///
    /// This invalidates any `BlockArgument` or `Value` handles that were
    /// obtained for the erased argument. Using such handles after this call
    /// is undefined behavior.
    unsafe fn erase_argument(&self, index: usize) {
        unsafe { mlirBlockEraseArgument(self.to_raw(), index as u32) }
    }

    /// Returns the number of successors.
    fn successor_count(&self) -> usize {
        unsafe { mlirBlockGetNumSuccessors(self.to_raw()) as usize }
    }

    /// Returns a successor block at a position.
    fn successor(&self, index: usize) -> Result<BlockRef<'c, 'a>, Error> {
        if index < self.successor_count() {
            Ok(unsafe { BlockRef::from_raw(mlirBlockGetSuccessor(self.to_raw(), index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "block successor",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns the number of predecessors.
    fn predecessor_count(&self) -> usize {
        unsafe { mlirBlockGetNumPredecessors(self.to_raw()) as usize }
    }

    /// Returns a predecessor block at a position.
    fn predecessor(&self, index: usize) -> Result<BlockRef<'c, 'a>, Error> {
        if index < self.predecessor_count() {
            Ok(unsafe {
                BlockRef::from_raw(mlirBlockGetPredecessor(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "block predecessor",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns a next block in a region.
    fn next_in_region(&self) -> Option<BlockRef<'c, 'a>> {
        unsafe { BlockRef::from_option_raw(mlirBlockGetNextInRegion(self.to_raw())) }
    }
}
