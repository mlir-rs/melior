mod utility;

use melior::ir::{Block, Location, Type, ValueLike, block::BlockLike, operation::OperationLike};
use utility::*;

melior_macro::dialect! {
    name: "same_type_test",
    files: ["macro/tests/ods_include/same_operands_result_type.td"],
}

#[test]
fn same_operands_result_type_has_result() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let i32_type = Type::parse(&context, "i32").unwrap();

    let block = Block::new(&[(i32_type, location), (i32_type, location)]);
    let lhs = block.argument(0).unwrap().into();
    let rhs = block.argument(1).unwrap().into();

    let op = same_type_test::add(&context, lhs, rhs, location);

    assert_eq!(op.as_operation().result_count(), 1);
    assert_eq!(op.result().unwrap().r#type(), i32_type,);
}

#[test]
fn variadic_first_operand_infers_result_type() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let i32_type = Type::parse(&context, "i32").unwrap();

    let block = Block::new(&[(i32_type, location), (i32_type, location)]);
    let a = block.argument(0).unwrap().into();
    let b = block.argument(1).unwrap().into();

    let op = same_type_test::sum(&context, &[a, b], location);

    assert_eq!(op.as_operation().result_count(), 1);
    assert_eq!(op.result().unwrap().r#type(), i32_type);
}

#[test]
fn tensor_type_infers_result_type() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let tensor_type = Type::parse(&context, "tensor<4xi32>").unwrap();

    let block = Block::new(&[(tensor_type, location), (tensor_type, location)]);
    let lhs = block.argument(0).unwrap().into();
    let rhs = block.argument(1).unwrap().into();

    let op = same_type_test::tensor_add(&context, lhs, rhs, location);

    assert_eq!(op.as_operation().result_count(), 1);
    assert_eq!(op.result().unwrap().r#type(), tensor_type);
}

#[test]
#[should_panic]
fn variadic_first_operand_empty_panics() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let _ = same_type_test::sum(&context, &[], location);
}
