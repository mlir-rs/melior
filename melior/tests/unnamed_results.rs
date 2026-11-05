//! Tests for builders and accessors generated for unnamed ODS results.
#![cfg(feature = "ods-dialects")]

use melior::{
    Context,
    ir::{
        Block, BlockLike, Location, Type, ValueLike, operation::OperationLike, r#type::IntegerType,
    },
};

melior_macro::dialect! {
    name: "unnamed_results",
    files: ["tests/ods_include/unnamed_results.td"],
    include_directory_env_vars: ["CARGO_MANIFEST_DIR"],
}

fn create_context() -> Context {
    let context = Context::new();

    context.set_allow_unregistered_dialects(true);

    context
}

#[test]
fn build_single_unnamed_result() {
    let context = create_context();
    let location = Location::unknown(&context);
    let r#type = IntegerType::new(&context, 32).into();
    let block = Block::new(&[(r#type, location)]);

    let operation = unnamed_results::SingleOperation::builder(&context, location)
        .input(block.argument(0).unwrap().into())
        .result_0(r#type)
        .build();

    assert_eq!(operation.as_operation().result_count(), 1);
    assert_eq!(operation.result_0().unwrap().r#type(), r#type);

    // The generated convenience constructor takes required results first, then
    // operands.
    let operation = unnamed_results::single(
        &context,
        r#type,
        block.argument(0).unwrap().into(),
        location,
    );

    assert_eq!(operation.as_operation().result_count(), 1);
}

#[test]
fn build_named_and_unnamed_results() {
    let context = create_context();
    let location = Location::unknown(&context);
    let i32_type: Type = IntegerType::new(&context, 32).into();
    let i64_type: Type = IntegerType::new(&context, 64).into();

    let operation = unnamed_results::MixedOperation::builder(&context, location)
        .first(i32_type)
        .result_1(i64_type)
        .build();

    assert_eq!(operation.as_operation().result_count(), 2);
    assert_eq!(operation.first().unwrap().r#type(), i32_type);
    assert_eq!(operation.result_1().unwrap().r#type(), i64_type);
}

#[test]
fn variadic_accessor_excludes_trailing_unnamed_result() {
    let context = create_context();
    let location = Location::unknown(&context);
    let i32_type: Type = IntegerType::new(&context, 32).into();
    let i64_type: Type = IntegerType::new(&context, 64).into();

    let operation = unnamed_results::VariadicOperation::builder(&context, location)
        .values(&[i32_type, i32_type])
        .result_1(i64_type)
        .build();

    assert_eq!(operation.as_operation().result_count(), 3);
    assert_eq!(operation.values().count(), 2);
    assert_eq!(operation.result_1().unwrap().r#type(), i64_type);
}

#[test]
fn infer_unnamed_result_type() {
    let context = create_context();
    let location = Location::unknown(&context);
    let r#type = IntegerType::new(&context, 32).into();
    let block = Block::new(&[(r#type, location)]);

    let operation = unnamed_results::InferOperation::builder(&context, location)
        .input(block.argument(0).unwrap().into())
        .build();

    assert_eq!(operation.as_operation().result_count(), 1);
    assert_eq!(operation.result_0().unwrap().r#type(), r#type);
}
