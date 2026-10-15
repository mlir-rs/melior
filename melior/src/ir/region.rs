mod region_like;

pub use self::region_like::RegionLike;
use super::{Block, BlockRef};
use mlir_sys::{
    MlirRegion, mlirBlockGetNextInRegion, mlirRegionCreate, mlirRegionDestroy, mlirRegionEqual,
};
use std::{
    marker::PhantomData,
    mem::{forget, transmute},
    ops::Deref,
};

/// A region.
#[derive(Debug)]
pub struct Region<'c> {
    raw: MlirRegion,
    _block: PhantomData<Block<'c>>,
}

impl Region<'_> {
    /// Creates a region.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirRegionCreate() },
            _block: Default::default(),
        }
    }

    /// Converts a region into a raw object.
    pub const fn into_raw(self) -> mlir_sys::MlirRegion {
        let region = self.raw;

        forget(self);

        region
    }
}

impl<'c: 'a, 'a> RegionLike<'c, 'a> for Region<'c> {
    fn to_raw(&self) -> MlirRegion {
        self.raw
    }
}

impl Default for Region<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Region<'_> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.raw) }
    }
}

impl PartialEq for Region<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for Region<'_> {}

/// A reference to a region.
#[derive(Clone, Copy, Debug)]
pub struct RegionRef<'c, 'a> {
    raw: MlirRegion,
    _region: PhantomData<&'a Region<'c>>,
}

impl RegionRef<'_, '_> {
    /// Creates a region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirRegion) -> Self {
        Self {
            raw,
            _region: Default::default(),
        }
    }

    /// Creates an optional region from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirRegion) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }
}

impl<'c: 'a, 'a> RegionLike<'c, 'a> for RegionRef<'c, 'a> {
    fn to_raw(&self) -> MlirRegion {
        self.raw
    }
}

impl<'c> Deref for RegionRef<'c, '_> {
    type Target = Region<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl PartialEq for RegionRef<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirRegionEqual(self.raw, other.raw) }
    }
}

impl Eq for RegionRef<'_, '_> {}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct RegionIterator<'c, 'a> {
    current: Option<BlockRef<'c, 'a>>,
}

impl<'c, 'a> RegionIterator<'c, 'a> {
    fn new(region: RegionRef<'c, 'a>) -> Self {
        Self {
            current: region.first_block(),
        }
    }
}

impl<'c, 'a> Iterator for RegionIterator<'c, 'a> {
    type Item = BlockRef<'c, 'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
            None => None,
            Some(op) => {
                self.current =
                    unsafe { BlockRef::from_option_raw(mlirBlockGetNextInRegion(op.to_raw())) };
                Some(op)
            }
        }
    }
}

impl<'c, 'a> IntoIterator for RegionRef<'c, 'a> {
    type Item = BlockRef<'c, 'a>;
    type IntoIter = RegionIterator<'c, 'a>;

    fn into_iter(self) -> Self::IntoIter {
        RegionIterator::new(self)
    }
}

impl<'c, 'a> IntoIterator for &'a Region<'c> {
    type Item = BlockRef<'c, 'a>;
    type IntoIter = RegionIterator<'c, 'a>;

    fn into_iter(self) -> Self::IntoIter {
        RegionIterator::new(unsafe { RegionRef::from_raw(self.to_raw()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Region::new();
    }

    #[test]
    fn first_block() {
        assert!(Region::new().first_block().is_none());
    }

    #[test]
    fn append_block() {
        let region = Region::new();
        let block = Block::new(&[]);

        region.append_block(block);

        assert!(region.first_block().is_some());
    }

    #[test]
    fn insert_block_after() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        region.insert_block_after(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn insert_block_before() {
        let region = Region::new();

        let block = region.append_block(Block::new(&[]));
        let block = region.insert_block_before(block, Block::new(&[]));

        assert_eq!(region.first_block(), Some(block));
    }

    #[test]
    fn equal() {
        let region = Region::new();

        assert_eq!(region, region);
    }

    #[test]
    fn not_equal() {
        assert_ne!(Region::new(), Region::new());
    }

    #[test]
    fn region_iterator() {
        let region = Region::new();

        let block1 = region.append_block(Block::new(&[]));
        let block2 = region.insert_block_after(block1, Block::new(&[]));

        let mut iter = region.into_iter();
        assert_eq!(iter.next(), Some(block1));
        assert_eq!(iter.next(), Some(block2));
        assert_eq!(iter.next(), None);
    }
}
