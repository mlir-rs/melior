use melior::{
    Context,
    ir::{Block, BlockLike, Location, Type, ValueLike},
};

fn main() {
    let context = Context::new();
    let block = Block::new(&[Type::index(&context)], &[Location::unknown(&context)]).unwrap();

    {
        let other_context = Context::new();

        block
            .argument(0)
            .unwrap()
            .set_type(Type::index(&other_context));
    }

    let _ = block.argument(0).unwrap().r#type().to_string();
}
