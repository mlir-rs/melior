use crate::{
    context::Context, ir_rewriter::RewriterBase, logical_result::LogicalResult,
    string_ref::StringRef,
};
use mlir_sys::{
    MlirFrozenRewritePatternSet, MlirOperation, MlirPatternRewriter, MlirRewritePattern,
    MlirRewritePatternCallbacks, MlirRewritePatternSet, mlirFreezeRewritePattern,
    mlirFrozenRewritePatternSetDestroy, mlirOpRewritePatternCreate, mlirPatternRewriterAsBase,
    mlirRewritePatternSetAdd, mlirRewritePatternSetCreate, mlirRewritePatternSetDestroy,
};
use std::{ffi::c_void, marker::PhantomData, mem::forget};

/// A set of rewrite patterns.
pub struct RewritePatternSet<'c> {
    raw: MlirRewritePatternSet,
    _context: PhantomData<&'c Context>,
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
    pub fn add(&self, pattern: RewritePattern) {
        unsafe { mlirRewritePatternSetAdd(self.raw, pattern.into_raw()) }
    }

    /// Freezes the pattern set into a frozen set. Consumes self.
    pub fn freeze(self) -> FrozenRewritePatternSet {
        let raw = unsafe { mlirFreezeRewritePattern(self.raw) };

        forget(self);

        FrozenRewritePatternSet { raw }
    }
}

impl Drop for RewritePatternSet<'_> {
    fn drop(&mut self) {
        unsafe { mlirRewritePatternSetDestroy(self.raw) }
    }
}

/// A frozen (immutable) rewrite pattern set.
pub struct FrozenRewritePatternSet {
    raw: MlirFrozenRewritePatternSet,
}

impl FrozenRewritePatternSet {
    /// Converts the frozen pattern set into a raw object, transferring
    /// ownership.
    pub fn into_raw(self) -> MlirFrozenRewritePatternSet {
        let raw = self.raw;

        forget(self);

        raw
    }
}

impl Drop for FrozenRewritePatternSet {
    fn drop(&mut self) {
        unsafe { mlirFrozenRewritePatternSetDestroy(self.raw) }
    }
}

/// A single rewrite pattern.
#[must_use = "add to a RewritePatternSet or resources will leak"]
pub struct RewritePattern {
    raw: MlirRewritePattern,
}

impl RewritePattern {
    /// Converts the pattern into a raw object, transferring ownership.
    pub fn into_raw(self) -> MlirRewritePattern {
        self.raw
    }
}

/// A pattern rewriter available inside a match-and-rewrite callback.
///
/// This is a non-owning reference; it must not outlive the callback invocation.
#[derive(Clone, Copy)]
pub struct PatternRewriter {
    raw: MlirPatternRewriter,
}

impl PatternRewriter {
    /// Creates a pattern rewriter from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirPatternRewriter) -> Self {
        Self { raw }
    }

    /// Returns the underlying rewriter base.
    pub fn as_rewriter_base(&self) -> RewriterBase<'_, '_> {
        unsafe { RewriterBase::from_raw(mlirPatternRewriterAsBase(self.raw)) }
    }
}

/// Creates an op rewrite pattern that matches operations with the given root
/// name.
///
/// The `callback` receives the pattern, the matched operation, and a pattern
/// rewriter. It should perform the rewrite and return `true` on success.
pub fn create_op_rewrite_pattern<F>(
    root_name: &str,
    benefit: u32,
    context: &Context,
    callback: F,
    generated_names: &[&str],
) -> RewritePattern
where
    F: FnMut(MlirRewritePattern, MlirOperation, MlirPatternRewriter) -> bool + 'static,
{
    unsafe extern "C" fn destruct<F>(user_data: *mut c_void) {
        unsafe {
            drop(Box::from_raw(user_data as *mut F));
        }
    }

    unsafe extern "C" fn match_and_rewrite<F>(
        pattern: MlirRewritePattern,
        op: MlirOperation,
        rewriter: MlirPatternRewriter,
        user_data: *mut c_void,
    ) -> mlir_sys::MlirLogicalResult
    where
        F: FnMut(MlirRewritePattern, MlirOperation, MlirPatternRewriter) -> bool,
    {
        let cb = unsafe { &mut *(user_data as *mut F) };
        let success = cb(pattern, op, rewriter);

        LogicalResult::from(success).to_raw()
    }

    let callbacks = MlirRewritePatternCallbacks {
        construct: None,
        destruct: Some(destruct::<F>),
        matchAndRewrite: Some(match_and_rewrite::<F>),
    };

    let user_data = Box::into_raw(Box::new(callback)) as *mut c_void;
    let root = StringRef::new(root_name);
    let mut generated: Vec<mlir_sys::MlirStringRef> = generated_names
        .iter()
        .map(|name| StringRef::new(name).to_raw())
        .collect();

    let raw = unsafe {
        mlirOpRewritePatternCreate(
            root.to_raw(),
            benefit,
            context.to_raw(),
            callbacks,
            user_data,
            generated.len(),
            generated.as_mut_ptr(),
        )
    };

    RewritePattern { raw }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        greedy_rewrite_driver::{GreedyRewriteDriverConfig, apply_patterns_and_fold_greedily},
        ir::{Location, Module},
        test::create_test_context,
    };

    #[test]
    fn new_pattern_set() {
        let context = Context::new();

        RewritePatternSet::new(&context);
    }

    #[test]
    fn freeze_pattern_set() {
        let context = Context::new();

        let set = RewritePatternSet::new(&context);
        let _frozen = set.freeze();
    }

    #[test]
    fn apply_frozen_patterns() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let patterns = RewritePatternSet::new(&context);
        let frozen = patterns.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, frozen, &config).is_ok());
    }

    #[test]
    fn create_and_add_op_rewrite_pattern() {
        let context = create_test_context();

        let pattern = create_op_rewrite_pattern(
            "arith.constant",
            1,
            &context,
            |_pattern, _op, _rewriter| true,
            &[],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn create_pattern_with_generated_names() {
        let context = create_test_context();

        let pattern = create_op_rewrite_pattern(
            "arith.constant",
            1,
            &context,
            |_pattern, _op, _rewriter| true,
            &["arith.addi"],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn apply_op_rewrite_pattern() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        let pattern = create_op_rewrite_pattern(
            "arith.constant",
            1,
            &context,
            |_pattern, _op, _rewriter| true,
            &[],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, frozen, &config).is_ok());
    }

    #[test]
    fn pattern_rewriter_as_rewriter_base() {
        use crate::{
            dialect::arith,
            ir::{BlockLike, RegionLike, Type, attribute::IntegerAttribute},
        };

        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let body = module.body();
        let location = Location::unknown(&context);

        let op = arith::constant(
            &context,
            IntegerAttribute::new(Type::index(&context), 0).into(),
            location,
        );

        body.append_operation(op);

        let pattern = create_op_rewrite_pattern(
            "arith.constant",
            1,
            &context,
            |_pattern, op, rewriter| {
                let rewriter = unsafe { PatternRewriter::from_raw(rewriter) };
                let base = rewriter.as_rewriter_base();
                let op = unsafe { crate::ir::OperationRef::from_raw(op) };

                base.erase_op(op);
                true
            },
            &[],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, frozen, &config).is_ok());
    }
}
