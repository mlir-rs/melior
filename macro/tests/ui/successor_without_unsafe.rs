use melior::{
    Context,
    dialect::ods::cf,
    ir::{Block, Location},
};

fn main() {
    let context = Context::new();
    let location = Location::unknown(&context);
    let block = Block::new(&[], &[]).unwrap();

    cf::br(&context, &[], &block, location);
    cf::BranchOperation::builder(&context, location)
        .dest_operands(&[])
        .dest(&block)
        .build();
}
