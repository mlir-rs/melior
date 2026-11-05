mod utility;

use melior::ir::{Location, Type, ValueLike, operation::OperationLike};
use utility::*;

melior_macro::dialect! {
    name: "result_test",
    files: ["macro/tests/ods_include/result.td"],
}

#[test]
fn attribute_sized_with_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let first_type = Type::parse(&context, "i8").unwrap();
    let optional_type = Type::parse(&context, "i16").unwrap();
    let other_types = [
        Type::parse(&context, "i32").unwrap(),
        Type::parse(&context, "i64").unwrap(),
    ];
    let operation = result_test::AttributeSizedOperation::builder(&context, location)
        .first(first_type)
        .optional(optional_type)
        .others(&other_types)
        .build();

    assert_eq!(operation.first().unwrap().r#type(), first_type);
    assert_eq!(operation.optional().unwrap().r#type(), optional_type);
    assert_eq!(
        operation
            .others()
            .unwrap()
            .map(|result| result.r#type())
            .collect::<Vec<_>>(),
        other_types
    );
    assert_eq!(operation.as_operation().result_count(), 4);
}

#[test]
fn attribute_sized_without_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let first_type = Type::parse(&context, "i8").unwrap();
    let other_types = [
        Type::parse(&context, "i32").unwrap(),
        Type::parse(&context, "i64").unwrap(),
    ];
    let operation = result_test::attribute_sized(&context, first_type, &other_types, location);

    assert_eq!(operation.first().unwrap().r#type(), first_type);
    assert!(operation.optional().is_err());
    assert_eq!(
        operation
            .others()
            .unwrap()
            .map(|result| result.r#type())
            .collect::<Vec<_>>(),
        other_types
    );
    assert_eq!(operation.as_operation().result_count(), 3);
}
