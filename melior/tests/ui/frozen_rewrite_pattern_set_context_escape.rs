use melior::{Context, RewritePatternSet};

fn main() {
    let _patterns = {
        let context = Context::new();

        RewritePatternSet::new(&context).freeze()
    };
}
