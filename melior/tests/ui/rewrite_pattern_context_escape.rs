use melior::{Context, RewritePattern, RewritePatternSet};

fn main() {
    let pattern = {
        let context = Context::new();

        RewritePattern::new("arith.constant", 1, &context, |_, _| false, &[]).unwrap()
    };

    let context = Context::new();
    RewritePatternSet::new(&context).add(pattern);
}
