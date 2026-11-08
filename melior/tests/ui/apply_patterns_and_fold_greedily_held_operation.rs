use melior::{
    Context, GreedyRewriteDriverConfig, RewritePatternSet, apply_patterns_and_fold_greedily,
    ir::{BlockLike, Module, RegionLike, operation::OperationLike},
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let mut module = Module::parse(
        &context,
        r#""foo.function"() ({ "foo.constant"() : () -> index }) : () -> ()"#,
    )
    .unwrap();
    let operation = module
        .body()
        .first_operation()
        .unwrap()
        .region(0)
        .unwrap()
        .first_block()
        .unwrap()
        .first_operation()
        .unwrap();

    apply_patterns_and_fold_greedily(
        &mut module,
        &RewritePatternSet::new(&context).freeze(),
        &GreedyRewriteDriverConfig::new(),
    )
    .unwrap();

    println!("{operation}");
}
