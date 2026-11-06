use melior::{
    Context,
    ir::{Location, Module, operation::OperationBuilder},
};

fn main() {
    let _module = {
        let context = Context::new();

        Module::from_operation(
            OperationBuilder::new("builtin.module", Location::unknown(&context))
                .build()
                .unwrap(),
        )
        .unwrap()
    };
}
