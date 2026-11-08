use melior::{
    Context, RewritePatternSet, walk_and_apply_patterns,
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

    unsafe {
        walk_and_apply_patterns(
            &mut module.as_operation_mut(),
            &RewritePatternSet::new(&context).freeze(),
        )
    };

    println!("{operation}");
}
