use melior::{
    Context,
    ir::{
        Location, Module, OperationRef,
        operation::{OperationLike, WalkOrder, WalkResult},
    },
};

fn main() {
    let context = Context::new();
    let module = Module::new(Location::unknown(&context));
    let mut operations = Vec::<OperationRef>::new();

    module.as_operation().walk(WalkOrder::PreOrder, |operation| {
        operations.push(operation);
        WalkResult::Advance
    });
}
