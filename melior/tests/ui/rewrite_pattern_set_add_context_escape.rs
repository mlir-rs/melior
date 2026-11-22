use melior::{Context, RewritePattern, RewritePatternSet};

fn main() {
    let context = Context::new();
    let set = RewritePatternSet::new(&context);

    {
        let other_context = Context::new();

        set.add(
            RewritePattern::new("arith.constant", 1, &other_context, |_, _| false, &[]).unwrap(),
        );
    }

    let _patterns = set.freeze();
}
