use melior::{
    Context, RewritePatternSet, walk_and_apply_patterns,
    ir::{BlockLike, Module, RegionLike, operation::OperationLike},
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let module = Module::parse(
        &context,
        r#""foo.function"() ({ "foo.a"() : () -> () }) : () -> ()"#,
    )
    .unwrap();
    let body = module.body();
    let operation = body
        .first_operation()
        .unwrap()
        .region(0)
        .unwrap()
        .first_block()
        .unwrap()
        .first_operation()
        .unwrap();

    walk_and_apply_patterns(
        &mut body.first_operation_mut().unwrap(),
        &RewritePatternSet::new(&context).freeze(),
    );

    println!("{operation}");
}
