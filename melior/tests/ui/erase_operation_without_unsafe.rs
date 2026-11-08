use melior::{
    Context, IrRewriter,
    ir::{Block, BlockLike, Location, operation::OperationBuilder},
};

fn main() {
    let context = Context::new();
    let block = Block::new(&[], &[]).unwrap();
    let operation = block.append_operation(
        OperationBuilder::new("foo.bar", Location::unknown(&context))
            .build()
            .unwrap(),
    );

    IrRewriter::new(&context)
        .as_rewriter_base()
        .erase_op(operation);

    println!("{operation}");
}
