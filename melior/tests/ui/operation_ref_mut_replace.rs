use melior::{
    Context,
    ir::{Location, Module, operation::OperationBuilder},
};
use std::mem::replace;

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    drop(replace(
        &mut *module.as_operation_mut(),
        OperationBuilder::new("foo.bar", location).build().unwrap(),
    ));
}
