use melior::{
    Context, GreedyRewriteDriverConfig, RewritePatternSet, apply_patterns_and_fold_greedily_with_op,
    ir::{
        Module,
        operation::{OperationMutLike, WalkOrder, WalkResult},
    },
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let mut module = Module::parse(
        &context,
        r#""foo.function"() ({ "foo.a"() : () -> () }) : () -> ()"#,
    )
    .unwrap();
    let patterns = RewritePatternSet::new(&context).freeze();

    module
        .as_operation_mut()
        .walk_mut(WalkOrder::PostOrder, |mut operation| {
            if let Some(mut parent) = operation.parent_operation_mut() {
                apply_patterns_and_fold_greedily_with_op(
                    &mut parent,
                    &patterns,
                    &GreedyRewriteDriverConfig::new(),
                )
                .unwrap();
            }

            WalkResult::Advance
        });
}
