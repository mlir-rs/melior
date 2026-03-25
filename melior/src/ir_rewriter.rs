use crate::{
    context::{Context, ContextRef},
    ir::{BlockLike, BlockRef, Operation, OperationRef, RegionLike, RegionRef, Value, ValueLike},
};
use mlir_sys::{
    MlirRewriterBase, MlirValue, mlirIRRewriterCreate, mlirIRRewriterCreateFromOp,
    mlirIRRewriterDestroy, mlirRewriterBaseCancelOpModification,
    mlirRewriterBaseClearInsertionPoint, mlirRewriterBaseClone, mlirRewriterBaseCloneRegionBefore,
    mlirRewriterBaseCloneWithoutRegions, mlirRewriterBaseEraseBlock, mlirRewriterBaseEraseOp,
    mlirRewriterBaseFinalizeOpModification, mlirRewriterBaseGetBlock, mlirRewriterBaseGetContext,
    mlirRewriterBaseGetInsertionBlock, mlirRewriterBaseInlineRegionBefore, mlirRewriterBaseInsert,
    mlirRewriterBaseMoveBlockBefore, mlirRewriterBaseMoveOpAfter, mlirRewriterBaseMoveOpBefore,
    mlirRewriterBaseReplaceAllUsesWith, mlirRewriterBaseReplaceOpWithOperation,
    mlirRewriterBaseReplaceOpWithValues, mlirRewriterBaseSetInsertionPointAfter,
    mlirRewriterBaseSetInsertionPointBefore, mlirRewriterBaseSetInsertionPointToEnd,
    mlirRewriterBaseSetInsertionPointToStart, mlirRewriterBaseStartOpModification,
};
use std::marker::PhantomData;

/// An IR rewriter. Owns the underlying rewriter object.
pub struct IrRewriter<'c> {
    raw: MlirRewriterBase,
    _context: PhantomData<&'c Context>,
}

impl<'c> IrRewriter<'c> {
    /// Creates an IR rewriter for the given context.
    pub fn new(context: &'c Context) -> Self {
        Self {
            raw: unsafe { mlirIRRewriterCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Creates an IR rewriter positioned before the given operation.
    pub fn from_op(op: OperationRef<'c, '_>) -> Self {
        Self {
            raw: unsafe { mlirIRRewriterCreateFromOp(op.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Returns the underlying rewriter base.
    pub fn as_rewriter_base(&self) -> RewriterBase<'c, '_> {
        unsafe { RewriterBase::from_raw(self.raw) }
    }
}

impl Drop for IrRewriter<'_> {
    fn drop(&mut self) {
        unsafe { mlirIRRewriterDestroy(self.raw) }
    }
}

/// A non-owning reference to a rewriter base.
#[derive(Clone, Copy)]
pub struct RewriterBase<'c, 'a> {
    raw: MlirRewriterBase,
    _context: PhantomData<&'c Context>,
    _reference: PhantomData<&'a ()>,
}

impl<'c, 'a> RewriterBase<'c, 'a> {
    /// Creates a rewriter base from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirRewriterBase) -> Self {
        Self {
            raw,
            _context: PhantomData,
            _reference: PhantomData,
        }
    }

    /// Returns the context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirRewriterBaseGetContext(self.raw)) }
    }

    /// Clears the insertion point.
    pub fn clear_insertion_point(&self) {
        unsafe { mlirRewriterBaseClearInsertionPoint(self.raw) }
    }

    /// Sets the insertion point before the given operation.
    pub fn set_insertion_point_before(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseSetInsertionPointBefore(self.raw, op.to_raw()) }
    }

    /// Sets the insertion point after the given operation.
    pub fn set_insertion_point_after(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseSetInsertionPointAfter(self.raw, op.to_raw()) }
    }

    /// Sets the insertion point to the start of the given block.
    pub fn set_insertion_point_to_start(&self, block: BlockRef) {
        unsafe { mlirRewriterBaseSetInsertionPointToStart(self.raw, block.to_raw()) }
    }

    /// Sets the insertion point to the end of the given block.
    pub fn set_insertion_point_to_end(&self, block: BlockRef) {
        unsafe { mlirRewriterBaseSetInsertionPointToEnd(self.raw, block.to_raw()) }
    }

    /// Returns the block the insertion point belongs to.
    pub fn insertion_block(&self) -> BlockRef<'c, '_> {
        unsafe { BlockRef::from_raw(mlirRewriterBaseGetInsertionBlock(self.raw)) }
    }

    /// Returns the current block.
    pub fn block(&self) -> BlockRef<'c, '_> {
        unsafe { BlockRef::from_raw(mlirRewriterBaseGetBlock(self.raw)) }
    }

    /// Inserts the operation at the current insertion point and returns a
    /// reference to it.
    pub fn insert(&self, op: Operation<'c>) -> OperationRef<'c, '_> {
        unsafe { OperationRef::from_raw(mlirRewriterBaseInsert(self.raw, op.into_raw())) }
    }

    /// Creates a deep copy of the operation.
    pub fn clone_op<'b>(&self, op: OperationRef<'c, 'b>) -> OperationRef<'c, 'b> {
        unsafe { OperationRef::from_raw(mlirRewriterBaseClone(self.raw, op.to_raw())) }
    }

    /// Creates a deep copy of the operation without its regions.
    pub fn clone_op_without_regions<'b>(&self, op: OperationRef<'c, 'b>) -> OperationRef<'c, 'b> {
        unsafe {
            OperationRef::from_raw(mlirRewriterBaseCloneWithoutRegions(self.raw, op.to_raw()))
        }
    }

    /// Clones the blocks of the region before the given block.
    pub fn clone_region_before(&self, region: RegionRef, before: BlockRef) {
        unsafe { mlirRewriterBaseCloneRegionBefore(self.raw, region.to_raw(), before.to_raw()) }
    }

    /// Moves the blocks of the region before the given block.
    pub fn inline_region_before(&self, region: RegionRef, before: BlockRef) {
        unsafe { mlirRewriterBaseInlineRegionBefore(self.raw, region.to_raw(), before.to_raw()) }
    }

    /// Replaces the results of the operation with the given values. Erases the
    /// op.
    pub fn replace_op_with_values(&self, op: OperationRef, values: &[Value]) {
        unsafe {
            mlirRewriterBaseReplaceOpWithValues(
                self.raw,
                op.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const MlirValue,
            )
        }
    }

    /// Replaces the operation with another operation. Erases the original op.
    pub fn replace_op_with_operation(&self, op: OperationRef, new_op: OperationRef) {
        unsafe { mlirRewriterBaseReplaceOpWithOperation(self.raw, op.to_raw(), new_op.to_raw()) }
    }

    /// Erases the operation. The operation must have no uses.
    pub fn erase_op(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseEraseOp(self.raw, op.to_raw()) }
    }

    /// Erases the block along with all its operations.
    pub fn erase_block(&self, block: BlockRef) {
        unsafe { mlirRewriterBaseEraseBlock(self.raw, block.to_raw()) }
    }

    /// Moves the operation immediately before the existing operation.
    pub fn move_op_before(&self, op: OperationRef, existing_op: OperationRef) {
        unsafe { mlirRewriterBaseMoveOpBefore(self.raw, op.to_raw(), existing_op.to_raw()) }
    }

    /// Moves the operation immediately after the existing operation.
    pub fn move_op_after(&self, op: OperationRef, existing_op: OperationRef) {
        unsafe { mlirRewriterBaseMoveOpAfter(self.raw, op.to_raw(), existing_op.to_raw()) }
    }

    /// Moves the block immediately before the existing block.
    pub fn move_block_before(&self, block: BlockRef, existing_block: BlockRef) {
        unsafe {
            mlirRewriterBaseMoveBlockBefore(self.raw, block.to_raw(), existing_block.to_raw())
        }
    }

    /// Signals the start of an in-place modification of the operation.
    pub fn start_op_modification(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseStartOpModification(self.raw, op.to_raw()) }
    }

    /// Signals the end of an in-place modification of the operation.
    pub fn finalize_op_modification(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseFinalizeOpModification(self.raw, op.to_raw()) }
    }

    /// Cancels a pending in-place modification of the operation.
    pub fn cancel_op_modification(&self, op: OperationRef) {
        unsafe { mlirRewriterBaseCancelOpModification(self.raw, op.to_raw()) }
    }

    /// Replaces all uses of `from` with `to`.
    pub fn replace_all_uses_with(&self, from: Value, to: Value) {
        unsafe { mlirRewriterBaseReplaceAllUsesWith(self.raw, from.to_raw(), to.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        dialect::arith,
        ir::{Location, Module, Type, attribute::IntegerAttribute},
        test::load_all_dialects,
    };

    #[test]
    fn new() {
        let context = Context::new();

        IrRewriter::new(&context);
    }

    #[test]
    fn set_insertion_point() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        base.set_insertion_point_to_start(body);
        base.set_insertion_point_to_end(body);
    }

    #[test]
    fn insert_and_erase() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        base.set_insertion_point_to_end(body);

        let location = Location::unknown(&context);
        let op = arith::constant(
            &context,
            IntegerAttribute::new(Type::index(&context), 0).into(),
            location,
        );

        let op_ref = base.insert(op);

        base.erase_op(op_ref);
    }

    #[test]
    fn move_op() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        base.set_insertion_point_to_end(body);

        let index_type = Type::index(&context);
        let location = Location::unknown(&context);

        let op1 = arith::constant(
            &context,
            IntegerAttribute::new(index_type, 1).into(),
            location,
        );

        let op2 = arith::constant(
            &context,
            IntegerAttribute::new(index_type, 2).into(),
            location,
        );

        let op1_ref = base.insert(op1);
        let op2_ref = base.insert(op2);

        base.move_op_before(op2_ref, op1_ref);
    }
}
