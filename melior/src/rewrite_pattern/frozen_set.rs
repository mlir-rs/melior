use crate::context::Context;
use core::{marker::PhantomData, mem::forget};
use mlir_sys::{MlirFrozenRewritePatternSet, mlirFrozenRewritePatternSetDestroy};

/// A frozen (immutable) rewrite pattern set.
pub struct FrozenRewritePatternSet<'c> {
    raw: MlirFrozenRewritePatternSet,
    _context: PhantomData<&'c Context>,
}

impl FrozenRewritePatternSet<'_> {
    /// Creates a frozen pattern set from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirFrozenRewritePatternSet) -> Self {
        Self {
            raw,
            _context: PhantomData,
        }
    }

    /// Converts the frozen pattern set into a raw object.
    pub const fn to_raw(&self) -> MlirFrozenRewritePatternSet {
        self.raw
    }

    /// Converts the frozen pattern set into a raw object, transferring
    /// ownership.
    pub const fn into_raw(self) -> MlirFrozenRewritePatternSet {
        let raw = self.raw;

        forget(self);

        raw
    }
}

impl Drop for FrozenRewritePatternSet<'_> {
    fn drop(&mut self) {
        unsafe { mlirFrozenRewritePatternSetDestroy(self.raw) }
    }
}
