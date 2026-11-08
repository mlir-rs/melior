use melior::{
    Context,
    ir::{Block, BlockLike, Location, operation::OperationBuilder},
};

fn main() {
    let context = Context::new();
    context.set_allow_unregistered_dialects(true);
    let location = Location::unknown(&context);

    let operation = {
        let block = Block::new(&[], &[]).unwrap();

        block.append_operation(OperationBuilder::new("foo.bar", location).build().unwrap())
    };

    let _ = operation.to_string();
}
