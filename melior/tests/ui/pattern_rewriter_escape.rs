use melior::{Context, PatternRewriter, RewritePattern};
use std::{cell::Cell, rc::Rc};

fn main() {
    let context = Context::new();
    let rewriter = Rc::new(Cell::new(None::<PatternRewriter>));
    let inner_rewriter = rewriter.clone();

    let _pattern = RewritePattern::new(
        "arith.constant",
        1,
        &context,
        move |_, rewriter| {
            inner_rewriter.set(Some(*rewriter));
            false
        },
        &[],
    )
    .unwrap();
}
