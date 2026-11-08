use crate::{
    Error,
    context::{Context, ContextRef},
    ir::{
        BlockLike, BlockRef, Location, Operation, OperationRef, RegionLike, RegionRef, Type, Value,
        ValueLike,
    },
};
use core::{cell::Cell, marker::PhantomData};
use mlir_sys::{
    MlirLocation, MlirRewriterBase, MlirType, MlirValue, mlirIRRewriterCreate,
    mlirIRRewriterCreateFromOp, mlirIRRewriterDestroy, mlirRewriterBaseCancelOpModification,
    mlirRewriterBaseClearInsertionPoint, mlirRewriterBaseClone, mlirRewriterBaseCloneRegionBefore,
    mlirRewriterBaseCloneWithoutRegions, mlirRewriterBaseCreateBlockBefore,
    mlirRewriterBaseEraseBlock, mlirRewriterBaseEraseOp, mlirRewriterBaseFinalizeOpModification,
    mlirRewriterBaseGetBlock, mlirRewriterBaseGetContext, mlirRewriterBaseGetInsertionBlock,
    mlirRewriterBaseGetOperationAfterInsertion, mlirRewriterBaseInlineBlockBefore,
    mlirRewriterBaseInlineRegionBefore, mlirRewriterBaseInsert, mlirRewriterBaseMergeBlocks,
    mlirRewriterBaseMoveBlockBefore, mlirRewriterBaseMoveOpAfter, mlirRewriterBaseMoveOpBefore,
    mlirRewriterBaseReplaceAllOpUsesWithOperation, mlirRewriterBaseReplaceAllOpUsesWithValueRange,
    mlirRewriterBaseReplaceAllUsesExcept, mlirRewriterBaseReplaceAllUsesWith,
    mlirRewriterBaseReplaceAllValueRangeUsesWith, mlirRewriterBaseReplaceOpUsesWithinBlock,
    mlirRewriterBaseReplaceOpWithOperation, mlirRewriterBaseReplaceOpWithValues,
    mlirRewriterBaseSetInsertionPointAfter, mlirRewriterBaseSetInsertionPointAfterValue,
    mlirRewriterBaseSetInsertionPointBefore, mlirRewriterBaseSetInsertionPointToEnd,
    mlirRewriterBaseSetInsertionPointToStart, mlirRewriterBaseStartOpModification,
};

/// An IR rewriter. Owns the underlying rewriter object.
pub struct IrRewriter<'c> {
    raw: MlirRewriterBase,
    _context: PhantomData<Cell<&'c Context>>,
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
    pub const fn as_rewriter_base(&self) -> RewriterBase<'c, '_> {
        unsafe { RewriterBase::from_raw(self.raw) }
    }
}

impl Drop for IrRewriter<'_> {
    fn drop(&mut self) {
        unsafe { mlirIRRewriterDestroy(self.raw) }
    }
}

/// A non-owning reference to a rewriter base.
// TODO Provide safe methods to erase IR, such as ones consuming exclusive
// handles.
#[derive(Clone, Copy)]
pub struct RewriterBase<'c, 'a> {
    raw: MlirRewriterBase,
    modified: Option<&'a Cell<bool>>,
    _context: PhantomData<Cell<&'c Context>>,
}

impl<'c, 'a> RewriterBase<'c, 'a> {
    /// Creates a rewriter base from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirRewriterBase) -> Self {
        unsafe { Self::from_raw_with_flag(raw, None) }
    }

    pub(crate) const unsafe fn from_raw_with_flag(
        raw: MlirRewriterBase,
        modified: Option<&'a Cell<bool>>,
    ) -> Self {
        Self {
            raw,
            modified,
            _context: PhantomData,
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
    pub fn set_insertion_point_before(&self, op: OperationRef<'c, '_>) {
        unsafe { mlirRewriterBaseSetInsertionPointBefore(self.raw, op.to_raw()) }
    }

    /// Sets the insertion point after the given operation.
    pub fn set_insertion_point_after(&self, op: OperationRef<'c, '_>) {
        unsafe { mlirRewriterBaseSetInsertionPointAfter(self.raw, op.to_raw()) }
    }

    /// Sets the insertion point to the start of the given block.
    pub fn set_insertion_point_to_start(&self, block: BlockRef<'c, '_>) {
        unsafe { mlirRewriterBaseSetInsertionPointToStart(self.raw, block.to_raw()) }
    }

    /// Sets the insertion point to the end of the given block.
    pub fn set_insertion_point_to_end(&self, block: BlockRef<'c, '_>) {
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
        self.mark_modified();
        unsafe { OperationRef::from_raw(mlirRewriterBaseInsert(self.raw, op.into_raw())) }
    }

    /// Creates a deep copy of the operation.
    pub fn clone_op<'b>(&self, op: OperationRef<'c, 'b>) -> OperationRef<'c, 'b> {
        self.mark_modified();
        unsafe { OperationRef::from_raw(mlirRewriterBaseClone(self.raw, op.to_raw())) }
    }

    /// Creates a deep copy of the operation without its regions.
    pub fn clone_op_without_regions<'b>(&self, op: OperationRef<'c, 'b>) -> OperationRef<'c, 'b> {
        self.mark_modified();
        unsafe {
            OperationRef::from_raw(mlirRewriterBaseCloneWithoutRegions(self.raw, op.to_raw()))
        }
    }

    /// Clones the blocks of the region before the given block.
    pub fn clone_region_before(&self, region: RegionRef<'c, '_>, before: BlockRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseCloneRegionBefore(self.raw, region.to_raw(), before.to_raw()) }
    }

    /// Moves the blocks of the region before the given block.
    pub fn inline_region_before(&self, region: RegionRef<'c, '_>, before: BlockRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseInlineRegionBefore(self.raw, region.to_raw(), before.to_raw()) }
    }

    /// Replaces the results of the operation with the given values. Erases the
    /// op.
    ///
    /// # Safety
    ///
    /// The values must be as many as the operation results, and neither the
    /// operation nor IR nested in it may be used after the replacement.
    pub unsafe fn replace_op_with_values(
        &self,
        op: OperationRef<'c, '_>,
        values: &[Value<'c, '_>],
    ) {
        self.mark_modified();
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
    ///
    /// # Safety
    ///
    /// The new operation must have as many results as the original one and
    /// must not be the original one or nested in it, and neither the original
    /// operation nor IR nested in it may be used after the replacement.
    pub unsafe fn replace_op_with_operation(
        &self,
        op: OperationRef<'c, '_>,
        new_op: OperationRef<'c, '_>,
    ) {
        self.mark_modified();
        unsafe { mlirRewriterBaseReplaceOpWithOperation(self.raw, op.to_raw(), new_op.to_raw()) }
    }

    /// Erases the operation.
    ///
    /// # Safety
    ///
    /// The operation must have no uses, and neither the operation nor IR nested
    /// in it may be used after the erasure.
    pub unsafe fn erase_op(&self, op: OperationRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseEraseOp(self.raw, op.to_raw()) }
    }

    /// Erases the block along with all its operations.
    ///
    /// # Safety
    ///
    /// The block must have no predecessors, values defined in it must have no
    /// uses outside of it, and neither the block nor IR nested in it may be
    /// used after the erasure.
    pub unsafe fn erase_block(&self, block: BlockRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseEraseBlock(self.raw, block.to_raw()) }
    }

    /// Moves the operation immediately before the existing operation.
    pub fn move_op_before(&self, op: OperationRef<'c, '_>, existing_op: OperationRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseMoveOpBefore(self.raw, op.to_raw(), existing_op.to_raw()) }
    }

    /// Moves the operation immediately after the existing operation.
    pub fn move_op_after(&self, op: OperationRef<'c, '_>, existing_op: OperationRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseMoveOpAfter(self.raw, op.to_raw(), existing_op.to_raw()) }
    }

    /// Moves the block immediately before the existing block.
    pub fn move_block_before(&self, block: BlockRef<'c, '_>, existing_block: BlockRef<'c, '_>) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseMoveBlockBefore(self.raw, block.to_raw(), existing_block.to_raw())
        }
    }

    /// Signals the start of an in-place modification of the operation.
    pub fn start_op_modification(&self, op: OperationRef<'c, '_>) {
        unsafe { mlirRewriterBaseStartOpModification(self.raw, op.to_raw()) }
    }

    /// Signals the end of an in-place modification of the operation.
    pub fn finalize_op_modification(&self, op: OperationRef<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseFinalizeOpModification(self.raw, op.to_raw()) }
    }

    /// Cancels a pending in-place modification of the operation.
    pub fn cancel_op_modification(&self, op: OperationRef<'c, '_>) {
        unsafe { mlirRewriterBaseCancelOpModification(self.raw, op.to_raw()) }
    }

    /// Replaces all uses of `from` with `to`.
    pub fn replace_all_uses_with(&self, from: Value<'c, '_>, to: Value<'c, '_>) {
        self.mark_modified();
        unsafe { mlirRewriterBaseReplaceAllUsesWith(self.raw, from.to_raw(), to.to_raw()) }
    }

    /// Sets the insertion point right after the definition of the given value.
    pub fn set_insertion_point_after_value(&self, value: Value<'c, '_>) {
        unsafe { mlirRewriterBaseSetInsertionPointAfterValue(self.raw, value.to_raw()) }
    }

    /// Returns the operation right after the insertion point, if any.
    pub fn operation_after_insertion(&self) -> Option<OperationRef<'c, '_>> {
        unsafe {
            OperationRef::from_option_raw(mlirRewriterBaseGetOperationAfterInsertion(self.raw))
        }
    }

    /// Creates a block with the given arguments before the given block.
    pub fn create_block_before(
        &self,
        before: BlockRef<'c, '_>,
        types: &[Type<'c>],
        locations: &[Location<'c>],
    ) -> Result<BlockRef<'c, '_>, Error> {
        if types.len() != locations.len() {
            return Err(Error::LocationCountMismatch {
                types: types.len(),
                locations: locations.len(),
            });
        }

        self.mark_modified();

        Ok(unsafe {
            BlockRef::from_raw(mlirRewriterBaseCreateBlockBefore(
                self.raw,
                before.to_raw(),
                types.len() as isize,
                types.as_ptr() as *const MlirType,
                locations.as_ptr() as *const MlirLocation,
            ))
        })
    }

    /// Inlines the source block before the given operation and erases it,
    /// replacing its arguments with the given values.
    ///
    /// # Safety
    ///
    /// The values must be as many as the source block arguments, the source
    /// block must have no predecessors and must not contain the operation, and
    /// the source block must not be used after the inlining.
    pub unsafe fn inline_block_before(
        &self,
        source: BlockRef<'c, '_>,
        op: OperationRef<'c, '_>,
        values: &[Value<'c, '_>],
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseInlineBlockBefore(
                self.raw,
                source.to_raw(),
                op.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const MlirValue,
            )
        }
    }

    /// Inlines the source block at the end of the destination block and erases
    /// it, replacing its arguments with the given values.
    ///
    /// # Safety
    ///
    /// The values must be as many as the source block arguments, the source
    /// block must have no predecessors and must differ from the destination
    /// block, and the source block must not be used after the merge.
    pub unsafe fn merge_blocks(
        &self,
        source: BlockRef<'c, '_>,
        destination: BlockRef<'c, '_>,
        values: &[Value<'c, '_>],
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseMergeBlocks(
                self.raw,
                source.to_raw(),
                destination.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const MlirValue,
            )
        }
    }

    /// Replaces all uses of the `from` values with the corresponding `to`
    /// values.
    pub fn replace_all_value_range_uses_with(
        &self,
        from: &[Value<'c, '_>],
        to: &[Value<'c, '_>],
    ) -> Result<(), Error> {
        if from.len() != to.len() {
            return Err(Error::ValueCountMismatch {
                from: from.len(),
                to: to.len(),
            });
        }

        self.mark_modified();

        unsafe {
            mlirRewriterBaseReplaceAllValueRangeUsesWith(
                self.raw,
                from.len() as isize,
                from.as_ptr() as *const MlirValue,
                to.as_ptr() as *const MlirValue,
            )
        }

        Ok(())
    }

    /// Replaces all uses of the operation results with the given values.
    pub fn replace_all_op_uses_with_values(
        &self,
        from: OperationRef<'c, '_>,
        to: &[Value<'c, '_>],
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseReplaceAllOpUsesWithValueRange(
                self.raw,
                from.to_raw(),
                to.len() as isize,
                to.as_ptr() as *const MlirValue,
            )
        }
    }

    /// Replaces all uses of the results of an operation with the results of
    /// another.
    pub fn replace_all_op_uses_with_operation(
        &self,
        from: OperationRef<'c, '_>,
        to: OperationRef<'c, '_>,
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseReplaceAllOpUsesWithOperation(self.raw, from.to_raw(), to.to_raw())
        }
    }

    /// Replaces uses of the operation results with the given values within a
    /// block.
    pub fn replace_op_uses_within_block(
        &self,
        op: OperationRef<'c, '_>,
        values: &[Value<'c, '_>],
        block: BlockRef<'c, '_>,
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseReplaceOpUsesWithinBlock(
                self.raw,
                op.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const MlirValue,
                block.to_raw(),
            )
        }
    }

    /// Replaces all uses of `from` with `to`, except uses by the excepted
    /// operation.
    pub fn replace_all_uses_except(
        &self,
        from: Value<'c, '_>,
        to: Value<'c, '_>,
        excepted_user: OperationRef<'c, '_>,
    ) {
        self.mark_modified();
        unsafe {
            mlirRewriterBaseReplaceAllUsesExcept(
                self.raw,
                from.to_raw(),
                to.to_raw(),
                excepted_user.to_raw(),
            )
        }
    }

    fn mark_modified(&self) {
        if let Some(modified) = self.modified {
            modified.set(true);
        }
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

        unsafe { base.erase_op(op_ref) };
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

    #[test]
    fn insertion_point_after_value() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        base.set_insertion_point_to_end(body);

        let op = arith::constant(
            &context,
            IntegerAttribute::new(Type::index(&context), 0).into(),
            Location::unknown(&context),
        );

        let op_ref = base.insert(op);

        base.set_insertion_point_after_value(op_ref.result(0).unwrap().into());

        assert!(base.operation_after_insertion().is_none());

        base.set_insertion_point_before(op_ref);

        assert!(base.operation_after_insertion().is_some());
    }

    #[test]
    fn replace_all_op_uses() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        base.set_insertion_point_to_end(body);

        let index_type = Type::index(&context);
        let location = Location::unknown(&context);

        let op1 = base.insert(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 1).into(),
            location,
        ));
        let op2 = base.insert(arith::constant(
            &context,
            IntegerAttribute::new(index_type, 2).into(),
            location,
        ));
        let sum = base.insert(arith::addi(
            op1.result(0).unwrap().into(),
            op1.result(0).unwrap().into(),
            location,
        ));

        base.replace_all_op_uses_with_operation(op1, op2);
        base.replace_all_op_uses_with_values(op2, &[op1.result(0).unwrap().into()]);
        base.replace_all_uses_except(
            op1.result(0).unwrap().into(),
            op2.result(0).unwrap().into(),
            sum,
        );
        base.replace_all_value_range_uses_with(
            &[op2.result(0).unwrap().into()],
            &[op1.result(0).unwrap().into()],
        )
        .unwrap();
        base.replace_op_uses_within_block(op1, &[op2.result(0).unwrap().into()], body);
    }

    #[test]
    fn replace_all_value_range_uses_with_mismatched_counts() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();

        base.set_insertion_point_to_end(module.body());

        let op = base.insert(arith::constant(
            &context,
            IntegerAttribute::new(Type::index(&context), 0).into(),
            Location::unknown(&context),
        ));

        assert_eq!(
            base.replace_all_value_range_uses_with(&[op.result(0).unwrap().into()], &[]),
            Err(Error::ValueCountMismatch { from: 1, to: 0 })
        );
    }

    #[test]
    fn create_block_before() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let r#type = Type::index(&context);

        let block = base
            .create_block_before(module.body(), &[r#type], &[Location::unknown(&context)])
            .unwrap();

        assert_eq!(block.argument_count(), 1);
        assert_eq!(block.argument(0).unwrap().r#type(), r#type);
    }

    #[test]
    fn create_block_before_with_mismatched_locations() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);

        assert_eq!(
            rewriter
                .as_rewriter_base()
                .create_block_before(module.body(), &[Type::index(&context)], &[])
                .unwrap_err(),
            Error::LocationCountMismatch {
                types: 1,
                locations: 0
            }
        );
    }

    #[test]
    fn block_surgery() {
        let context = Context::new();
        load_all_dialects(&context);

        let module = Module::new(Location::unknown(&context));
        let rewriter = IrRewriter::new(&context);
        let base = rewriter.as_rewriter_base();
        let body = module.body();

        let block = base.create_block_before(body, &[], &[]).unwrap();

        base.set_insertion_point_to_end(block);

        let op = arith::constant(
            &context,
            IntegerAttribute::new(Type::index(&context), 0).into(),
            Location::unknown(&context),
        );

        base.insert(op);

        unsafe { base.merge_blocks(block, body, &[]) };
    }
}
