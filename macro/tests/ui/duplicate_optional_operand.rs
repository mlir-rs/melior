use melior::{
    Context,
    dialect::ods::amdgpu,
    ir::{Block, BlockLike, Location, Type},
};

fn main() {
    let context = Context::new();
    let location = Location::unknown(&context);
    let r#type = Type::index(&context);
    let block = Block::new(&[(r#type, location)]);

    amdgpu::RawBufferLoadOperation::builder(&context, location)
        .value(r#type)
        .memref(block.argument(0).unwrap().into())
        .indices(&[block.argument(0).unwrap().into()])
        .sgpr_offset(block.argument(0).unwrap().into())
        .sgpr_offset(block.argument(0).unwrap().into())
        .build();
}
