use melior::{
    Context,
    ir::{
        Location, Type, ValueLike,
        operation::{OperationBuilder, OperationLike},
    },
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);

    let result = {
        let operation = OperationBuilder::new("foo.bar", Location::unknown(&context))
            .add_results(&[Type::index(&context)])
            .build()
            .unwrap();

        operation.result(0).unwrap()
    };

    let _ = result.r#type().to_string();
}
