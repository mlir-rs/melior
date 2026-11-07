use melior::{
    Context,
    ir::{Location, Module, attribute::Attribute, operation::OperationMutLike},
};

fn main() {
    let context = Context::new();
    let mut module = Module::new(Location::unknown(&context));
    let mut operation = module.as_operation_mut();
    let mut other = operation;

    operation.set_attribute("foo", Attribute::unit(&context));
    other.set_attribute("bar", Attribute::unit(&context));
}
