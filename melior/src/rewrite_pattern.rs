use crate::{
    context::Context, ir::OperationRef, ir_rewriter::RewriterBase, logical_result::LogicalResult,
    string_ref::StringRef,
};
use core::{cell::Cell, ffi::c_void, marker::PhantomData, mem::forget};
use mlir_sys::{
    MlirFrozenRewritePatternSet, MlirOperation, MlirPatternRewriter, MlirRewritePattern,
    MlirRewritePatternCallbacks, MlirRewritePatternSet, mlirFreezeRewritePattern,
    mlirFrozenRewritePatternSetDestroy, mlirOpRewritePatternCreate, mlirPatternRewriterAsBase,
    mlirRewritePatternSetAdd, mlirRewritePatternSetCreate, mlirRewritePatternSetDestroy,
};

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
        FrozenRewritePatternSet {
            raw: unsafe { mlirFreezeRewritePattern(self.raw) },
        }
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
    pub const fn into_raw(self) -> MlirRewritePattern {
        self.raw
    }
}

/// A pattern rewriter available inside a match-and-rewrite callback.
///
/// This is a non-owning reference; it must not outlive the callback invocation.
#[derive(Clone, Copy)]
pub struct PatternRewriter<'c, 'a> {
    raw: MlirPatternRewriter,
    modified: Option<&'a Cell<bool>>,
    _context: PhantomData<Cell<&'c Context>>,
}

impl<'c> PatternRewriter<'c, '_> {
    /// Creates a pattern rewriter from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirPatternRewriter) -> Self {
        Self {
            raw,
            modified: None,
            _context: PhantomData,
        }
    }

    /// Returns the underlying rewriter base.
    pub fn as_rewriter_base(&self) -> RewriterBase<'c, '_> {
        unsafe {
            RewriterBase::from_raw_with_flag(mlirPatternRewriterAsBase(self.raw), self.modified)
        }
    }
}

/// Creates an op rewrite pattern that matches operations with the given root
/// name.
///
/// The `callback` receives the matched operation and a pattern rewriter. It
/// should rewrite the operation and return `true`, or return `false` without
/// modifying the IR if the pattern does not match. The pattern is reported as
/// matched whenever the callback modifies the IR through the rewriter, even if
/// it returns `false`. Panics in the callback abort the process because they
/// might leave the IR partially rewritten.
pub fn create_op_rewrite_pattern<F>(
    root_name: &str,
    benefit: u32,
    context: &Context,
    callback: F,
    generated_names: &[&str],
) -> RewritePattern
where
    F: for<'c> FnMut(OperationRef<'c, '_>, &PatternRewriter<'c, '_>) -> bool + 'static,
{
    unsafe extern "C" fn destruct<F>(user_data: *mut c_void) {
        unsafe {
            drop(Box::from_raw(user_data as *mut F));
        }
    }

    unsafe extern "C" fn match_and_rewrite<F>(
        _pattern: MlirRewritePattern,
        op: MlirOperation,
        rewriter: MlirPatternRewriter,
        user_data: *mut c_void,
    ) -> mlir_sys::MlirLogicalResult
    where
        F: for<'c> FnMut(OperationRef<'c, '_>, &PatternRewriter<'c, '_>) -> bool,
    {
        let modified = Cell::new(false);

        LogicalResult::from(
            unsafe {
                (*(user_data as *mut F))(
                    OperationRef::from_raw(op),
                    &PatternRewriter {
                        raw: rewriter,
                        modified: Some(&modified),
                        _context: PhantomData,
                    },
                )
            } || modified.get(),
        )
        .to_raw()
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
    use alloc::rc::Rc;
    use indoc::indoc;

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

        assert!(apply_patterns_and_fold_greedily(&module, &frozen, &config).is_ok());
    }

    #[test]
    fn create_and_add_op_rewrite_pattern() {
        let context = create_test_context();

        let pattern =
            create_op_rewrite_pattern("arith.constant", 1, &context, |_op, _rewriter| false, &[]);

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
            |_op, _rewriter| false,
            &["arith.addi"],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn apply_op_rewrite_pattern() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));

        let pattern =
            create_op_rewrite_pattern("arith.constant", 1, &context, |_op, _rewriter| false, &[]);

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, &frozen, &config).is_ok());
    }

    #[test]
    fn pattern_rewriter_as_rewriter_base() {
        let context = create_test_context();
        let module = Module::parse(
            &context,
            indoc!(
                "
                func.func private @foo()

                func.func @bar() {
                    func.call @foo() : () -> ()
                    return
                }
                "
            ),
        )
        .unwrap();

        let pattern = create_op_rewrite_pattern(
            "func.call",
            1,
            &context,
            |op, rewriter| {
                unsafe { rewriter.as_rewriter_base().erase_op(op) };

                true
            },
            &[],
        );

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, &frozen, &config).is_ok());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn fall_back_to_next_pattern() {
        let context = create_test_context();
        let module = Module::parse(
            &context,
            indoc!(
                "
                func.func private @foo()

                func.func @bar() {
                    func.call @foo() : () -> ()
                    return
                }
                "
            ),
        )
        .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(create_op_rewrite_pattern(
            "func.call",
            2,
            &context,
            |_op, _rewriter| false,
            &[],
        ));
        set.add(create_op_rewrite_pattern(
            "func.call",
            1,
            &context,
            |op, rewriter| {
                unsafe { rewriter.as_rewriter_base().erase_op(op) };

                true
            },
            &[],
        ));

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, &frozen, &config).is_ok());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn report_modification_as_match() {
        let context = create_test_context();
        let module = Module::parse(
            &context,
            indoc!(
                "
                func.func private @foo()

                func.func @bar() {
                    func.call @foo() : () -> ()
                    return
                }
                "
            ),
        )
        .unwrap();
        let fallen_back = Rc::new(Cell::new(false));

        let set = RewritePatternSet::new(&context);
        set.add(create_op_rewrite_pattern(
            "func.call",
            2,
            &context,
            |op, rewriter| {
                unsafe { rewriter.as_rewriter_base().erase_op(op) };

                false
            },
            &[],
        ));
        set.add(create_op_rewrite_pattern(
            "func.call",
            1,
            &context,
            {
                let fallen_back = fallen_back.clone();

                move |_op, _rewriter| {
                    fallen_back.set(true);

                    false
                }
            },
            &[],
        ));

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, &frozen, &config).is_ok());
        assert!(!fallen_back.get());
        insta::assert_snapshot!(module.as_operation());
    }
}
