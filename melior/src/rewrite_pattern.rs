mod frozen_set;
mod rewriter;
mod set;

pub use self::{
    frozen_set::FrozenRewritePatternSet, rewriter::PatternRewriter, set::RewritePatternSet,
};
use crate::{
    Error, context::Context, ir::OperationRef, logical_result::LogicalResult, string_ref::StringRef,
};
use core::{ffi::c_void, marker::PhantomData};
use mlir_sys::{
    MlirOperation, MlirPatternRewriter, MlirRewritePattern, MlirRewritePatternCallbacks,
    mlirOpRewritePatternCreate,
};

/// A single rewrite pattern.
#[must_use = "add to a RewritePatternSet or resources will leak"]
pub struct RewritePattern<'c> {
    raw: MlirRewritePattern,
    _context: PhantomData<&'c Context>,
}

impl<'c> RewritePattern<'c> {
    /// Creates a rewrite pattern that matches operations with the given root
    /// name.
    ///
    /// The `callback` receives the matched operation and a pattern rewriter. It
    /// should rewrite the operation and return `true`, or return `false`
    /// without modifying the IR if the pattern does not match. Panics in the
    /// callback abort the process because they might leave the IR partially
    /// rewritten.
    ///
    /// A benefit of `u16::MAX` is rejected because MLIR reserves it for
    /// patterns impossible to match.
    pub fn new<F>(
        root_name: &str,
        benefit: u16,
        context: &'c Context,
        callback: F,
        generated_names: &[&str],
    ) -> Result<Self, Error>
    where
        F: for<'a> FnMut(OperationRef<'a, '_>, &PatternRewriter<'a, '_>) -> bool + 'static,
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
            LogicalResult::from(unsafe {
                (*(user_data as *mut F))(
                    OperationRef::from_raw(op),
                    &PatternRewriter::from_raw(rewriter),
                )
            })
            .to_raw()
        }

        if benefit == u16::MAX {
            return Err(Error::PatternBenefitOutOfBounds(benefit));
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
                benefit.into(),
                context.to_raw(),
                callbacks,
                user_data,
                generated.len(),
                generated.as_mut_ptr(),
            )
        };

        Ok(Self {
            raw,
            _context: Default::default(),
        })
    }

    /// Converts the pattern into a raw object, transferring ownership.
    pub const fn into_raw(self) -> MlirRewritePattern {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        greedy_rewrite_driver::{GreedyRewriteDriverConfig, apply_patterns_and_fold_greedily},
        ir::{Location, Module},
        test::create_test_context,
    };
    use alloc::rc::Rc;
    use core::cell::RefCell;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        let context = create_test_context();

        let pattern =
            RewritePattern::new("arith.constant", 1, &context, |_op, _rewriter| false, &[])
                .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn new_with_generated_names() {
        let context = create_test_context();

        let pattern = RewritePattern::new(
            "arith.constant",
            1,
            &context,
            |_op, _rewriter| false,
            &["arith.addi"],
        )
        .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn new_with_largest_benefit() {
        let context = create_test_context();

        let pattern = RewritePattern::new(
            "arith.constant",
            u16::MAX - 1,
            &context,
            |_op, _rewriter| false,
            &[],
        )
        .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(pattern);
    }

    #[test]
    fn fail_to_create_with_out_of_bounds_benefit() {
        let context = create_test_context();

        assert!(matches!(
            RewritePattern::new(
                "arith.constant",
                u16::MAX,
                &context,
                |_op, _rewriter| false,
                &[],
            ),
            Err(Error::PatternBenefitOutOfBounds(u16::MAX))
        ));
    }

    #[test]
    fn drop_callback_with_out_of_bounds_benefit() {
        let context = create_test_context();
        let matched = Rc::new(false);

        assert!(
            RewritePattern::new(
                "arith.constant",
                u16::MAX,
                &context,
                {
                    let matched = matched.clone();

                    move |_op, _rewriter| *matched
                },
                &[],
            )
            .is_err()
        );
        assert_eq!(Rc::strong_count(&matched), 1);
    }

    #[test]
    fn apply_frozen_patterns() {
        let context = create_test_context();
        let mut module = Module::new(Location::unknown(&context));
        let patterns = RewritePatternSet::new(&context);
        let frozen = patterns.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&mut module, &frozen, &config).is_ok());
    }

    #[test]
    fn apply_pattern() {
        let context = create_test_context();
        let mut module = Module::new(Location::unknown(&context));

        let pattern =
            RewritePattern::new("arith.constant", 1, &context, |_op, _rewriter| false, &[])
                .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&mut module, &frozen, &config).is_ok());
    }

    #[test]
    fn fall_back_to_next_pattern() {
        let context = create_test_context();
        let mut module = Module::parse(
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
        set.add(
            RewritePattern::new("func.call", 2, &context, |_op, _rewriter| false, &[]).unwrap(),
        );
        set.add(
            RewritePattern::new(
                "func.call",
                1,
                &context,
                |op, rewriter| {
                    rewriter.as_rewriter_base().erase_op(op);

                    true
                },
                &[],
            )
            .unwrap(),
        );

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&mut module, &frozen, &config).is_ok());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn apply_patterns_in_benefit_order() {
        let context = create_test_context();
        let mut module = Module::parse(
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
        let benefits = Rc::new(RefCell::new(vec![]));
        let set = RewritePatternSet::new(&context);

        for benefit in [1, 256, u16::MAX - 1] {
            set.add(
                RewritePattern::new(
                    "func.call",
                    benefit,
                    &context,
                    {
                        let benefits = benefits.clone();

                        move |_op, _rewriter| {
                            benefits.borrow_mut().push(benefit);

                            false
                        }
                    },
                    &[],
                )
                .unwrap(),
            );
        }

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&mut module, &frozen, &config).is_ok());
        assert_eq!(*benefits.borrow(), [u16::MAX - 1, 256, 1]);
    }
}
