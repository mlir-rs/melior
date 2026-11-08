use melior::ir::{Block, Region, RegionLike};

fn main() {
    let block = {
        let region = Region::new();

        region.append_block(Block::new(&[], &[]).unwrap())
    };

    let _ = block.to_string();
}
