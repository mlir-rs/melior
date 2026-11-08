use melior::{
    Context,
    ir::{Block, BlockLike, Location, Type, ValueLike},
};

fn main() {
    let block = Block::new(&[], &[]).unwrap();

    {
        let context = Context::new();

        block.add_argument(Type::index(&context), Location::unknown(&context));
    }

    let _ = block.argument(0).unwrap().r#type().to_string();
}
