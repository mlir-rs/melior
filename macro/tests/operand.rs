#![expect(missing_docs)]

mod utility;

use melior::ir::{Block, Location, Type, ValueLike, block::BlockLike, operation::OperationLike};
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
    let block = Block::new(&[r#type, r#type], &[location, location]).unwrap();
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
    assert_eq!(operation.as_operation().result_count(), 1);
    assert_eq!(operation.res().unwrap().r#type(), r#type);
}

#[test]
fn variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
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
    assert_eq!(operation.as_operation().result_count(), 1);
    assert_eq!(operation.res().unwrap().r#type(), r#type);
}

#[test]
fn empty_variadic_after_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type], &[location]).unwrap();
    let operation = operand_test::variadic(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        &[],
        location,
    );

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(operation.others().count(), 0);
    assert_eq!(operation.as_operation().operand_count(), 1);
}

#[test]
fn variadic_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::variadic_before_single(
        &context,
        r#type,
        &[
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        ],
        block.argument(2).unwrap().into(),
        location,
    );

    assert_eq!(
        operation.others().next(),
        Some(block.argument(0).unwrap().into())
    );
    assert_eq!(
        operation.others().nth(1),
        Some(block.argument(1).unwrap().into())
    );
    assert_eq!(operation.others().count(), 2);
    assert_eq!(operation.last().unwrap(), block.argument(2).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 3);
}

#[test]
fn empty_variadic_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type], &[location]).unwrap();
    let operation = operand_test::variadic_before_single(
        &context,
        r#type,
        &[],
        block.argument(0).unwrap().into(),
        location,
    );

    assert_eq!(operation.others().count(), 0);
    assert_eq!(operation.last().unwrap(), block.argument(0).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 1);
}

#[test]
fn optional_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type], &[location, location]).unwrap();
    let operation = operand_test::OptionalBeforeSingleOperation::builder(&context, location)
        .res(r#type)
        .optional(block.argument(0).unwrap().into())
        .last(block.argument(1).unwrap().into())
        .build();

    assert_eq!(
        operation.optional().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(operation.last().unwrap(), block.argument(1).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 2);
}

#[test]
fn absent_optional_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type], &[location]).unwrap();
    let operation = operand_test::optional_before_single(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        location,
    );

    assert!(operation.optional().is_err());
    assert_eq!(operation.last().unwrap(), block.argument(0).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 1);
}

#[test]
fn maybe_optional_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type], &[location, location]).unwrap();
    let operation = operand_test::OptionalBeforeSingleOperation::builder(&context, location)
        .res(r#type)
        .maybe_optional(Some(block.argument(0).unwrap().into()))
        .last(block.argument(1).unwrap().into())
        .build();

    assert_eq!(
        operation.optional().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(operation.last().unwrap(), block.argument(1).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 2);
}

#[test]
fn absent_maybe_optional_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type], &[location]).unwrap();
    let operation = operand_test::OptionalBeforeSingleOperation::builder(&context, location)
        .res(r#type)
        .maybe_optional(None)
        .last(block.argument(0).unwrap().into())
        .build();

    assert!(operation.optional().is_err());
    assert_eq!(operation.last().unwrap(), block.argument(0).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 1);
}

#[test]
fn attribute_sized_variadic_named_sizes() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::attribute_sized(
        &context,
        r#type,
        &[block.argument(0).unwrap().into()],
        &[
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ],
        location,
    );

    assert_eq!(
        operation.sizes().unwrap().next(),
        Some(block.argument(0).unwrap().into())
    );
    assert_eq!(operation.sizes().unwrap().count(), 1);
    assert_eq!(
        operation.others().unwrap().next(),
        Some(block.argument(1).unwrap().into())
    );
    assert_eq!(
        operation.others().unwrap().nth(1),
        Some(block.argument(2).unwrap().into())
    );
    assert_eq!(operation.others().unwrap().count(), 2);
    assert_eq!(operation.as_operation().operand_count(), 3);
}

#[test]
fn attribute_sized_with_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(
        &[r#type, r#type, r#type, r#type],
        &[location, location, location, location],
    )
    .unwrap();
    let operation = operand_test::AttributeSizedOptionalOperation::builder(&context, location)
        .res(r#type)
        .first(block.argument(0).unwrap().into())
        .optional(block.argument(1).unwrap().into())
        .others(&[
            block.argument(2).unwrap().into(),
            block.argument(3).unwrap().into(),
        ])
        .build();

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        operation.optional().unwrap(),
        block.argument(1).unwrap().into()
    );
    assert_eq!(
        operation.others().unwrap().next(),
        Some(block.argument(2).unwrap().into())
    );
    assert_eq!(
        operation.others().unwrap().nth(1),
        Some(block.argument(3).unwrap().into())
    );
    assert_eq!(operation.others().unwrap().count(), 2);
    assert_eq!(operation.as_operation().operand_count(), 4);
}

#[test]
fn attribute_sized_without_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::attribute_sized_optional(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        &[
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ],
        location,
    );

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert!(operation.optional().is_err());
    assert_eq!(
        operation.others().unwrap().next(),
        Some(block.argument(1).unwrap().into())
    );
    assert_eq!(
        operation.others().unwrap().nth(1),
        Some(block.argument(2).unwrap().into())
    );
    assert_eq!(operation.others().unwrap().count(), 2);
    assert_eq!(operation.as_operation().operand_count(), 3);
}

#[test]
fn attribute_sized_with_maybe_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(
        &[r#type, r#type, r#type, r#type],
        &[location, location, location, location],
    )
    .unwrap();
    let operation = operand_test::AttributeSizedOptionalOperation::builder(&context, location)
        .res(r#type)
        .first(block.argument(0).unwrap().into())
        .maybe_optional(Some(block.argument(1).unwrap().into()))
        .others(&[
            block.argument(2).unwrap().into(),
            block.argument(3).unwrap().into(),
        ])
        .build();

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        operation.optional().unwrap(),
        block.argument(1).unwrap().into()
    );
    assert_eq!(
        operation.others().unwrap().collect::<Vec<_>>(),
        [
            block.argument(2).unwrap().into(),
            block.argument(3).unwrap().into(),
        ]
    );
    assert_eq!(operation.as_operation().operand_count(), 4);
}

#[test]
fn attribute_sized_without_maybe_optional() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::AttributeSizedOptionalOperation::builder(&context, location)
        .res(r#type)
        .first(block.argument(0).unwrap().into())
        .maybe_optional(None)
        .others(&[
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ])
        .build();

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert!(operation.optional().is_err());
    assert_eq!(
        operation.others().unwrap().collect::<Vec<_>>(),
        [
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ]
    );
    assert_eq!(operation.as_operation().operand_count(), 3);
}

#[test]
fn same_size() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(
        &[r#type, r#type, r#type, r#type],
        &[location, location, location, location],
    )
    .unwrap();
    let operation = operand_test::same_size(
        &context,
        r#type,
        &[
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        ],
        &[
            block.argument(2).unwrap().into(),
            block.argument(3).unwrap().into(),
        ],
        location,
    );

    assert_eq!(
        operation.lhs().collect::<Vec<_>>(),
        [
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        ]
    );
    assert_eq!(
        operation.rhs().collect::<Vec<_>>(),
        [
            block.argument(2).unwrap().into(),
            block.argument(3).unwrap().into(),
        ]
    );
    assert_eq!(operation.as_operation().operand_count(), 4);
}

#[test]
fn empty_same_size() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let operation = operand_test::same_size(
        &context,
        Type::parse(&context, "i32").unwrap(),
        &[],
        &[],
        location,
    );

    assert_eq!(operation.lhs().count(), 0);
    assert_eq!(operation.rhs().count(), 0);
    assert_eq!(operation.as_operation().operand_count(), 0);
}

#[test]
fn same_size_with_singles() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(
        &[r#type, r#type, r#type, r#type, r#type, r#type, r#type],
        &[
            location, location, location, location, location, location, location,
        ],
    )
    .unwrap();
    let operation = operand_test::same_size_with_singles(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        &[
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ],
        block.argument(3).unwrap().into(),
        &[
            block.argument(4).unwrap().into(),
            block.argument(5).unwrap().into(),
        ],
        block.argument(6).unwrap().into(),
        location,
    );

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        operation.lhs().collect::<Vec<_>>(),
        [
            block.argument(1).unwrap().into(),
            block.argument(2).unwrap().into(),
        ]
    );
    assert_eq!(
        operation.middle().unwrap(),
        block.argument(3).unwrap().into()
    );
    assert_eq!(
        operation.rhs().collect::<Vec<_>>(),
        [
            block.argument(4).unwrap().into(),
            block.argument(5).unwrap().into(),
        ]
    );
    assert_eq!(operation.last().unwrap(), block.argument(6).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 7);
}

#[test]
fn empty_same_size_with_singles() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::same_size_with_singles(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        &[],
        block.argument(1).unwrap().into(),
        &[],
        block.argument(2).unwrap().into(),
        location,
    );

    assert_eq!(
        operation.first().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(operation.lhs().count(), 0);
    assert_eq!(
        operation.middle().unwrap(),
        block.argument(1).unwrap().into()
    );
    assert_eq!(operation.rhs().count(), 0);
    assert_eq!(operation.last().unwrap(), block.argument(2).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 3);
}

#[test]
fn unnamed() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type], &[location, location]).unwrap();
    let operation = operand_test::unnamed(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        block.argument(1).unwrap().into(),
        location,
    );

    assert_eq!(
        operation.operand_0().unwrap(),
        block.argument(0).unwrap().into()
    );
    assert_eq!(
        operation.named().unwrap(),
        block.argument(1).unwrap().into()
    );
    assert_eq!(operation.as_operation().operand_count(), 2);
}

#[test]
fn unnamed_variadic_before_single() {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);

    let location = Location::unknown(&context);

    let r#type = Type::parse(&context, "i32").unwrap();
    let block = Block::new(&[r#type, r#type, r#type], &[location, location, location]).unwrap();
    let operation = operand_test::UnnamedVariadicOperation::builder(&context, location)
        .res(r#type)
        .operand_0(&[
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        ])
        .last(block.argument(2).unwrap().into())
        .build();

    assert_eq!(
        operation.operand_0().collect::<Vec<_>>(),
        [
            block.argument(0).unwrap().into(),
            block.argument(1).unwrap().into(),
        ]
    );
    assert_eq!(operation.last().unwrap(), block.argument(2).unwrap().into());
    assert_eq!(operation.as_operation().operand_count(), 3);
}
