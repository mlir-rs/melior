mod utility;

use melior::ir::{Block, Location, Type};
use utility::*;

melior_macro::dialect! {
    name: "operand_test",
    td_file: "macro/tests/ods_include/operand.td",
}

#[test]
fn simple() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[(r#type, location), (r#type, location)]);
    let op = operand_test::simple(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        block.argument(1).unwrap().into(),
        location,
    );

    assert_eq!(op.lhs(&context).unwrap(), block.argument(0).unwrap().into());
    assert_eq!(op.rhs(&context).unwrap(), block.argument(1).unwrap().into());
    assert_eq!(op.operation().operand_count(), 2);
}

#[test]
fn variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[(r#type, location), (r#type, location), (r#type, location)]);
    let op = operand_test::variadic(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        &[
            block.argument(2).unwrap().into(),
            block.argument(1).unwrap().into(),
        ],
        location,
    );

    assert_eq!(
        op.first(&context).unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        op.others(&context).next(),
        Some(block.argument(2).unwrap().into())
    );
    assert_eq!(
        op.others(&context).nth(1),
        Some(block.argument(1).unwrap().into())
    );
    assert_eq!(op.operation().operand_count(), 3);
    assert_eq!(op.others(&context).count(), 2);
}
