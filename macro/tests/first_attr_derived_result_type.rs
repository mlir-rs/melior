mod utility;

use melior::ir::{
    Location, Type, ValueLike,
    attribute::{Attribute, TypeAttribute},
    operation::OperationLike,
};
use utility::*;

melior_macro::dialect! {
    name: "first_attr_test",
    files: ["macro/tests/ods_include/first_attr_derived_result_type.td"],
}

#[test]
fn type_attr_derives_result_type() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let i32_type = Type::parse(&context, "i32").unwrap();
    let type_attr = TypeAttribute::new(i32_type);

    let op = first_attr_test::r#const(&context, type_attr, location);

    assert_eq!(op.as_operation().result_count(), 1);
    assert_eq!(op.result().unwrap().r#type(), i32_type);
}

#[test]
fn non_type_attr_derives_result_type_from_attr_type() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);
    let i32_type = Type::parse(&context, "i32").unwrap();
    // "42 : i32" is an IntegerAttr whose .r#type() returns i32.
    let attr = Attribute::parse(&context, "42 : i32").unwrap();

    let op = first_attr_test::from_attr(&context, attr, location);

    assert_eq!(op.as_operation().result_count(), 1);
    assert_eq!(op.result().unwrap().r#type(), i32_type);
}
