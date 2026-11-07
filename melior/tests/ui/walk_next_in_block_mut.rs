use melior::{
    Context,
    ir::{
        BlockLike, Location, Module,
        operation::{OperationBuilder, OperationLike, OperationMutLike, WalkOrder, WalkResult},
    },
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let location = Location::unknown(&context);
    let module = Module::new(location);

    for name in ["foo.bar", "foo.baz"] {
        module
            .body()
            .append_operation(OperationBuilder::new(name, location).build().unwrap());
    }

    module.as_operation().walk(WalkOrder::PreOrder, |operation| {
        if let Some(mut next) = operation.next_in_block_mut() {
            let _ = next.remove_attribute("foo");
        }

        WalkResult::Advance
    });
}
