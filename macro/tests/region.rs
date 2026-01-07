mod utility;

use melior::ir::{Block, Location, Region, RegionLike};
use utility::*;

melior_macro::dialect! {
    name: "region_test",
    files: ["macro/tests/ods_include/region.td"],
}

#[test]
fn single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let region = Region::new();
    region.append_block(Block::new(&[]));
    let operation = region_test::single(&context, region, location);

    assert!(operation.default_region().unwrap().first_block().is_some());
}

#[test]
fn variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let one_operation = {
        let block = Block::new(&[]);
        let regions = (Region::new(), Region::new(), Region::new());
        regions.1.append_block(block);
        region_test::variadic(&context, regions.0, vec![regions.1, regions.2], location)
    };

    let other_operation = {
        let block = Block::new(&[]);
        let (r1, r2, r3) = (Region::new(), Region::new(), Region::new());
        r2.append_block(block);
        region_test::VariadicOperation::builder(&context, location)
            .default_region(r1)
            .other_regions(vec![r2, r3])
            .build()
    };

    assert_eq!(
        one_operation.as_operation().to_string(),
        other_operation.as_operation().to_string()
    );

    assert!(
        one_operation
            .default_region()
            .unwrap()
            .first_block()
            .is_none()
    );
    assert_eq!(one_operation.other_regions().count(), 2);
    assert!(
        one_operation
            .other_regions()
            .next()
            .unwrap()
            .first_block()
            .is_some()
    );
    assert!(
        one_operation
            .other_regions()
            .nth(1)
            .unwrap()
            .first_block()
            .is_none()
    );
}
