use melior::{
    Context,
    dialect::ods::bufferization,
    ir::{Block, BlockLike, Location, Type},
};

fn main() {
    let context = Context::new();
    let location = Location::unknown(&context);
    let r#type = Type::index(&context);
    let block = Block::new(&[(r#type, location)]);

    bufferization::AllocTensorOperation::builder(&context, location)
        .result(r#type)
        .dynamic_sizes(&[])
        .size_hint(block.argument(0).unwrap().into())
        .copy(block.argument(0).unwrap().into())
        .build();
}
