use super::{FrozenRewritePatternSet, RewritePattern};
use crate::context::Context;
use core::{cell::Cell, marker::PhantomData};
use mlir_sys::{
    MlirRewritePatternSet, mlirFreezeRewritePattern, mlirRewritePatternSetAdd,
    mlirRewritePatternSetCreate, mlirRewritePatternSetDestroy,
};

/// A set of rewrite patterns.
pub struct RewritePatternSet<'c> {
    raw: MlirRewritePatternSet,
    _context: PhantomData<Cell<&'c Context>>,
}

impl<'c> RewritePatternSet<'c> {
    /// Creates a rewrite pattern set.
    pub fn new(context: &'c Context) -> Self {
        Self {
            raw: unsafe { mlirRewritePatternSetCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Adds a pattern to the set. The pattern's ownership is transferred.
    pub fn add(&self, pattern: RewritePattern<'c>) {
        unsafe { mlirRewritePatternSetAdd(self.raw, pattern.into_raw()) }
    }

    /// Freezes the pattern set into a frozen set. Consumes self.
    pub fn freeze(self) -> FrozenRewritePatternSet<'c> {
        unsafe { FrozenRewritePatternSet::from_raw(mlirFreezeRewritePattern(self.raw)) }
    }
}

impl Drop for RewritePatternSet<'_> {
    fn drop(&mut self) {
        unsafe { mlirRewritePatternSetDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let context = Context::new();

        RewritePatternSet::new(&context);
    }

    #[test]
    fn freeze() {
        let context = Context::new();

        let set = RewritePatternSet::new(&context);
        let _frozen = set.freeze();
    }
}
