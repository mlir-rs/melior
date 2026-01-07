mod utility;

use melior::ir::{Block, Location, Type, block::BlockLike, operation::OperationLike};
use utility::*;

melior_macro::dialect! {
    name: "operand_test",
    files: ["macro/tests/ods_include/operand.td"],
}

#[test]
fn simple() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[(r#type, location), (r#type, location)]);
    let operation = operand_test::simple(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        block.argument(1).unwrap().into(),
        location,
    );

    assert_eq!(operation.lhs().unwrap(), block.argument(0).unwrap().into());
    assert_eq!(operation.rhs().unwrap(), block.argument(1).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 2);
}

#[test]
fn variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[(r#type, location), (r#type, location), (r#type, location)]);
    let operation = operand_test::variadic(
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
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        operation.others().next(),
        Some(block.argument(2).unwrap().into())
    );
    assert_eq!(
        operation.others().nth(1),
        Some(block.argument(1).unwrap().into())
    );
    assert_eq!(operation.as_operation().operand_count(), 3);
    assert_eq!(operation.others().count(), 2);
}
