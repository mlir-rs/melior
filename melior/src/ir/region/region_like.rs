use crate::ir::{Block, BlockRef};
use mlir_sys::{
    mlirRegionAppendOwnedBlock, mlirRegionGetFirstBlock, mlirRegionInsertOwnedBlockAfter,
    mlirRegionInsertOwnedBlockBefore, MlirRegion,
};

/// A region-like trait.
pub trait RegionLike<'c, 'a>: 'a {
    /// Converts a region into a raw object.
    fn to_raw(&self) -> MlirRegion;

    /// Returns the first block in a region.
    fn first_block(&self) -> Option<BlockRef<'c, 'a>> {
        unsafe {
            let block = mlirRegionGetFirstBlock(self.to_raw());

            if block.ptr.is_null() {
                None
            } else {
                Some(BlockRef::from_raw(block))
            }
        }
    }

    /// Inserts a block after another block.
    fn insert_block_after(&self, one: BlockRef<'c, 'a>, other: Block<'c>) -> BlockRef<'c, 'a> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockAfter(self.to_raw(), one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Inserts a block before another block.
    fn insert_block_before(&self, one: BlockRef<'c, 'a>, other: Block<'c>) -> BlockRef<'c, 'a> {
        unsafe {
            let r#ref = BlockRef::from_raw(other.to_raw());

            mlirRegionInsertOwnedBlockBefore(self.to_raw(), one.to_raw(), other.into_raw());

            r#ref
        }
    }

    /// Appends a block.
    fn append_block(&self, block: Block<'c>) -> BlockRef<'c, 'a> {
        unsafe {
            let r#ref = BlockRef::from_raw(block.to_raw());

            mlirRegionAppendOwnedBlock(self.to_raw(), block.into_raw());

            r#ref
        }
    }
}
