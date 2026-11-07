use melior::{
    Context,
    ir::{BlockLike, Location, Module, operation::OperationBuilder},
};

fn main() {
    let context = Context::new();
    let module = Module::new(Location::unknown(&context));

    {
        let other_context = Context::new();

        module.body().append_operation(
            OperationBuilder::new("foo.bar", Location::unknown(&other_context))
                .build()
                .unwrap(),
        );
    }

    let _ = module.as_operation().to_string();
}
