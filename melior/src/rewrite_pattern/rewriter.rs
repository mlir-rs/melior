use crate::{context::Context, ir_rewriter::RewriterBase};
use core::{cell::Cell, marker::PhantomData};
use mlir_sys::{MlirPatternRewriter, mlirPatternRewriterAsBase};

/// A pattern rewriter available inside a match-and-rewrite callback.
///
/// This is a non-owning reference; it must not outlive the callback invocation.
#[derive(Clone, Copy)]
pub struct PatternRewriter<'c, 'a> {
    raw: MlirPatternRewriter,
    _context: PhantomData<Cell<&'c Context>>,
    _reference: PhantomData<&'a MlirPatternRewriter>,
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
            _context: PhantomData,
            _reference: PhantomData,
        }
    }

    /// Returns the underlying rewriter base.
    pub fn as_rewriter_base(&self) -> RewriterBase<'c, '_> {
        unsafe { RewriterBase::from_raw(mlirPatternRewriterAsBase(self.raw)) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        greedy_rewrite_driver::{GreedyRewriteDriverConfig, apply_patterns_and_fold_greedily},
        ir::Module,
        rewrite_pattern::{RewritePattern, RewritePatternSet},
        test::create_test_context,
    };
    use indoc::indoc;

    #[test]
    fn as_rewriter_base() {
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

        let pattern = RewritePattern::new(
            "func.call",
            1,
            &context,
            |op, rewriter| {
                rewriter.as_rewriter_base().erase_op(op);

                true
            },
            &[],
        )
        .unwrap();

        let set = RewritePatternSet::new(&context);
        set.add(pattern);

        let frozen = set.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&mut module, &frozen, &config).is_ok());
        insta::assert_snapshot!(module.as_operation());
    }
}
