use std::{ffi::c_void, fmt::Display};

use mlir_sys::{
    MlirOperation, MlirWalkOrder_MlirWalkPostOrder, MlirWalkOrder_MlirWalkPreOrder, MlirWalkResult,
    MlirWalkResult_MlirWalkResultAdvance, MlirWalkResult_MlirWalkResultInterrupt,
    MlirWalkResult_MlirWalkResultSkip, mlirOperationDump, mlirOperationGetAttribute,
    mlirOperationGetAttributeByName, mlirOperationGetBlock, mlirOperationGetContext,
    mlirOperationGetLocation, mlirOperationGetName, mlirOperationGetNextInBlock,
    mlirOperationGetNumAttributes, mlirOperationGetNumOperands, mlirOperationGetNumRegions,
    mlirOperationGetNumResults, mlirOperationGetNumSuccessors, mlirOperationGetOperand,
    mlirOperationGetParentOperation, mlirOperationGetRegion, mlirOperationGetResult,
    mlirOperationGetSuccessor, mlirOperationPrintWithFlags, mlirOperationRemoveAttributeByName,
    mlirOperationRemoveFromParent, mlirOperationSetAttributeByName, mlirOperationVerify,
    mlirOperationWalk,
};

use crate::{
    ContextRef, Error, StringRef,
    ir::{Attribute, AttributeLike, BlockRef, Identifier, Location, RegionRef, Value},
};

use super::{
    OperationPrintingFlags, OperationRef, OperationRefMut, OperationResult, print_string_callback,
};

/// Order in which to traverse an operation tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum WalkOrder {
    /// Visit the operation before its nested regions.
    PreOrder = MlirWalkOrder_MlirWalkPreOrder,
    /// Visit the operation after its nested regions.
    PostOrder = MlirWalkOrder_MlirWalkPostOrder,
}

/// Control flow action returned by the walk callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum WalkResult {
    /// Continue into this operation’s children.
    Advance = MlirWalkResult_MlirWalkResultAdvance,
    /// Terminate the entire walk immediately.
    Interrupt = MlirWalkResult_MlirWalkResultInterrupt,
    /// Don’t visit this operation’s children, but keep walking siblings.
    Skip = MlirWalkResult_MlirWalkResultSkip,
}

pub trait OperationLike<'c: 'a, 'a>: Display + 'a {
    /// Converts a value into a raw value.
    fn to_raw(&self) -> MlirOperation;

    /// Returns a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirOperationGetContext(self.to_raw())) }
    }

    /// Returns a name.
    fn name(&self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirOperationGetName(self.to_raw())) }
    }

    /// Returns a block.
    // TODO Store lifetime of block in operations, or create another type like
    // `AppendedOperationRef`?
    fn block(&self) -> Option<BlockRef<'c, 'a>> {
        unsafe { BlockRef::from_option_raw(mlirOperationGetBlock(self.to_raw())) }
    }

    /// Returns the number of operands.
    fn operand_count(&self) -> usize {
        unsafe { mlirOperationGetNumOperands(self.to_raw()) as usize }
    }

    /// Returns the operand at a position.
    fn operand(&self, index: usize) -> Result<Value<'c, 'a>, Error> {
        if index < self.operand_count() {
            Ok(unsafe { Value::from_raw(mlirOperationGetOperand(self.to_raw(), index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation operand",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all operands.
    fn operands(&self) -> impl Iterator<Item = Value<'c, 'a>> {
        (0..self.operand_count()).map(|index| self.operand(index).expect("valid operand index"))
    }

    /// Returns the number of results.
    fn result_count(&self) -> usize {
        unsafe { mlirOperationGetNumResults(self.to_raw()) as usize }
    }

    /// Returns a result at a position.
    fn result(&self, index: usize) -> Result<OperationResult<'c, 'a>, Error> {
        if index < self.result_count() {
            Ok(unsafe {
                OperationResult::from_raw(mlirOperationGetResult(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "operation result",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all results.
    fn results(&self) -> impl Iterator<Item = OperationResult<'c, 'a>> {
        (0..self.result_count()).map(|index| self.result(index).expect("valid result index"))
    }

    /// Returns the number of regions.
    fn region_count(&self) -> usize {
        unsafe { mlirOperationGetNumRegions(self.to_raw()) as usize }
    }

    /// Returns a region at a position.
    fn region(&self, index: usize) -> Result<RegionRef<'c, 'a>, Error> {
        if index < self.region_count() {
            Ok(unsafe {
                RegionRef::from_raw(mlirOperationGetRegion(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all regions.
    fn regions(&self) -> impl Iterator<Item = RegionRef<'c, 'a>> {
        (0..self.region_count()).map(move |index| self.region(index).expect("valid result index"))
    }

    /// Gets the location of the operation.
    fn location(&self) -> Location<'c> {
        unsafe { Location::from_raw(mlirOperationGetLocation(self.to_raw())) }
    }

    /// Returns the number of successors.
    fn successor_count(&self) -> usize {
        unsafe { mlirOperationGetNumSuccessors(self.to_raw()) as usize }
    }

    /// Returns a successor at a position.
    fn successor(&self, index: usize) -> Result<BlockRef<'c, 'a>, Error> {
        if index < self.successor_count() {
            Ok(unsafe {
                BlockRef::from_raw(mlirOperationGetSuccessor(self.to_raw(), index as isize))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "successor",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all successors.
    fn successors(&self) -> impl Iterator<Item = BlockRef<'c, 'a>> {
        (0..self.successor_count())
            .map(|index| self.successor(index).expect("valid successor index"))
    }

    /// Returns the number of attributes.
    fn attribute_count(&self) -> usize {
        unsafe { mlirOperationGetNumAttributes(self.to_raw()) as usize }
    }

    /// Returns a attribute at a position.
    fn attribute_at(&self, index: usize) -> Result<(Identifier<'c>, Attribute<'c>), Error> {
        if index < self.attribute_count() {
            let named_attribute =
                unsafe { mlirOperationGetAttribute(self.to_raw(), index as isize) };

            Ok((
                unsafe { Identifier::from_raw(named_attribute.name) },
                unsafe { Attribute::from_raw(named_attribute.attribute) },
            ))
        } else {
            Err(Error::PositionOutOfBounds {
                name: "attribute",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns all attributes.
    fn attributes(&self) -> impl Iterator<Item = (Identifier<'c>, Attribute<'c>)> + '_ {
        (0..self.attribute_count())
            .map(|index| self.attribute_at(index).expect("valid attribute index"))
    }

    /// Returns a attribute with the given name.
    fn attribute(&self, name: &str) -> Result<Attribute<'c>, Error> {
        unsafe {
            Attribute::from_option_raw(mlirOperationGetAttributeByName(
                self.to_raw(),
                StringRef::new(name).to_raw(),
            ))
        }
        .ok_or_else(|| Error::AttributeNotFound(name.into()))
    }

    /// Checks if the operation has a attribute with the given name.
    fn has_attribute(&self, name: &str) -> bool {
        self.attribute(name).is_ok()
    }

    /// Returns a reference to the next operation in the same block.
    fn next_in_block(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirOperationGetNextInBlock(self.to_raw())) }
    }

    /// Returns a mutable reference to the next operation in the same block.
    fn next_in_block_mut(&self) -> Option<OperationRefMut<'c, 'a>> {
        unsafe { OperationRefMut::from_option_raw(mlirOperationGetNextInBlock(self.to_raw())) }
    }

    /// Returns a reference to the previous operation in the same block.
    fn previous_in_block(&self) -> Option<OperationRef<'c, 'a>> {
        todo!("mlirOperationGetPrevInBlock is not exposed in the C API")
    }

    /// Returns a reference to a parent operation.
    fn parent_operation(&self) -> Option<OperationRef<'c, 'a>> {
        unsafe { OperationRef::from_option_raw(mlirOperationGetParentOperation(self.to_raw())) }
    }

    /// Verifies an operation.
    fn verify(&self) -> bool {
        unsafe { mlirOperationVerify(self.to_raw()) }
    }

    /// Dumps an operation.
    fn dump(&self) {
        unsafe { mlirOperationDump(self.to_raw()) }
    }

    /// Prints an operation with flags.
    fn to_string_with_flags(&self, flags: OperationPrintingFlags) -> Result<String, Error> {
        let mut data = (String::new(), Ok::<_, Error>(()));

        unsafe {
            mlirOperationPrintWithFlags(
                self.to_raw(),
                flags.to_raw(),
                Some(print_string_callback),
                &mut data as *mut _ as *mut _,
            );
        }

        data.1?;

        Ok(data.0)
    }

    /// Walk this operation (and all nested operations) in either pre- or
    /// post-order.
    ///
    /// The closure is called once per operation; by returning
    /// `WalkResult::Advance`/`Skip`/`Interrupt` you control the traversal.
    fn walk<F>(&self, order: WalkOrder, mut callback: F)
    where
        F: for<'x, 'y> FnMut(OperationRef<'x, 'y>) -> WalkResult,
    {
        // trampoline from C to Rust
        unsafe extern "C" fn tramp<'c: 'a, 'a, F: FnMut(OperationRef<'c, 'a>) -> WalkResult>(
            operation: MlirOperation,
            data: *mut c_void,
        ) -> MlirWalkResult {
            let callback: &mut F = unsafe { &mut *(data as *mut F) };
            let operation = unsafe { OperationRef::from_raw(operation) };

            (callback)(operation) as _
        }

        unsafe {
            mlirOperationWalk(
                self.to_raw(),
                Some(tramp::<'c, 'a, F>),
                &mut callback as *mut _ as *mut _,
                order as _,
            );
        }
    }
}

pub trait OperationMutLike<'c: 'a, 'a>: OperationLike<'c, 'a> {
    /// Sets the attribute with the given name to the given attribute.
    fn set_attribute(&mut self, name: &str, attribute: Attribute<'c>) {
        unsafe {
            mlirOperationSetAttributeByName(
                self.to_raw(),
                StringRef::new(name).to_raw(),
                attribute.to_raw(),
            )
        }
    }

    /// Removes the attribute with the given name.
    fn remove_attribute(&mut self, name: &str) -> Result<(), Error> {
        unsafe { mlirOperationRemoveAttributeByName(self.to_raw(), StringRef::new(name).to_raw()) }
            .then_some(())
            .ok_or_else(|| Error::AttributeNotFound(name.into()))
    }

    /// Removes itself from a parent block.
    fn remove_from_parent(&mut self) {
        unsafe { mlirOperationRemoveFromParent(self.to_raw()) }
    }
}
