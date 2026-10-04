//! `enzyme` dialect.
//!
//! Builders for [Enzyme](https://enzyme.ad)'s automatic-differentiation dialect. The `enzyme`
//! dialect's verifiers, passes, and lowerings are provided by Enzyme's MLIR C API
//! (`enzymeRegisterDialectExtensions`), not by Melior. Operations built here parse, print, and
//! pass MLIR's generic verifier in any context with `set_allow_unregistered_dialects(true)`, but
//! are only semantically meaningful (checked by Enzyme's verifiers, lowerable by Enzyme's passes)
//! in a context where that dialect extension has been registered.

use crate::{
    Context,
    ir::{
        Attribute, Identifier, Location, Operation, Type, Value,
        attribute::{
            ArrayAttribute, BoolAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
    },
};

/// Activity states for `enzyme.autodiff`, `enzyme.fwddiff`, and `enzyme.jacobian` operands and
/// results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activity {
    Active,
    Dup,
    Const,
    DupNoNeed,
    ActiveNoNeed,
    ConstNoNeed,
}

impl Activity {
    const fn mnemonic(self) -> &'static str {
        match self {
            Self::Active => "enzyme_active",
            Self::Dup => "enzyme_dup",
            Self::Const => "enzyme_const",
            Self::DupNoNeed => "enzyme_dupnoneed",
            Self::ActiveNoNeed => "enzyme_activenoneed",
            Self::ConstNoNeed => "enzyme_constnoneed",
        }
    }
}

/// Creates an `#enzyme<activity ...>` attribute.
///
/// # Panics
///
/// Panics if `context` does not allow unregistered dialects (see
/// [`Context::set_allow_unregistered_dialects`](crate::Context::set_allow_unregistered_dialects))
/// and does not have the `enzyme` dialect registered.
pub fn activity_attribute(context: &Context, activity: Activity) -> Attribute<'_> {
    Attribute::parse(
        context,
        &format!("#enzyme<activity {}>", activity.mnemonic()),
    )
    .expect("`#enzyme<activity ...>` attributes are parseable")
}

/// Creates an array of `#enzyme<activity ...>` attributes.
///
/// # Panics
///
/// See [`activity_attribute`].
pub fn activity_array_attribute<'c>(
    context: &'c Context,
    activities: &[Activity],
) -> ArrayAttribute<'c> {
    let attributes = activities
        .iter()
        .map(|activity| activity_attribute(context, *activity))
        .collect::<Vec<_>>();

    ArrayAttribute::new(context, &attributes)
}

#[allow(clippy::too_many_arguments)]
fn diff_operation<'c>(
    name: &str,
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: ArrayAttribute<'c>,
    ret_activity: ArrayAttribute<'c>,
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new(name, location)
        .add_operands(operands)
        .add_results(result_types)
        .add_attributes(&[
            (Identifier::new(context, "fn"), function.into()),
            (Identifier::new(context, "activity"), activity.into()),
            (
                Identifier::new(context, "ret_activity"),
                ret_activity.into(),
            ),
            (
                Identifier::new(context, "width"),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), width).into(),
            ),
            (
                Identifier::new(context, "strong_zero"),
                BoolAttribute::new(context, strong_zero).into(),
            ),
        ])
        .build()
        .expect("valid operation")
}

/// Creates an `enzyme.autodiff` operation.
///
/// Computes the reverse-mode derivative of `function` with respect to its `enzyme_active` /
/// `enzyme_dup` operands.
#[allow(clippy::too_many_arguments)]
pub fn autodiff<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: ArrayAttribute<'c>,
    ret_activity: ArrayAttribute<'c>,
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_operation(
        "enzyme.autodiff",
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Creates an `enzyme.fwddiff` operation.
///
/// Computes the forward-mode (tangent) derivative of `function`.
#[allow(clippy::too_many_arguments)]
pub fn fwddiff<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: ArrayAttribute<'c>,
    ret_activity: ArrayAttribute<'c>,
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_operation(
        "enzyme.fwddiff",
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Creates an `enzyme.jacobian` operation.
///
/// Computes the full Jacobian of `function` with respect to its `enzyme_active` / `enzyme_dup`
/// operands.
#[allow(clippy::too_many_arguments)]
pub fn jacobian<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    activity: ArrayAttribute<'c>,
    ret_activity: ArrayAttribute<'c>,
    width: i64,
    strong_zero: bool,
    location: Location<'c>,
) -> Operation<'c> {
    diff_operation(
        "enzyme.jacobian",
        context,
        function,
        operands,
        result_types,
        activity,
        ret_activity,
        width,
        strong_zero,
        location,
    )
}

/// Creates an `enzyme.batch` operation.
///
/// Vectorizes a call to `function` over the leading `batch_shape` dimensions of its operands and
/// results.
pub fn batch<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    operands: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    batch_shape: &[i64],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("enzyme.batch", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_attributes(&[
            (Identifier::new(context, "fn"), function.into()),
            (
                Identifier::new(context, "batch_shape"),
                DenseI64ArrayAttribute::new(context, batch_shape).into(),
            ),
        ])
        .build()
        .expect("valid operation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        dialect::{DialectRegistry, func},
        ir::{
            Block, BlockLike, Module, Region, RegionLike, Type,
            attribute::{StringAttribute, TypeAttribute},
            operation::OperationLike,
            r#type::FunctionType,
        },
        utility::register_all_dialects,
    };

    fn create_test_context() -> Context {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);

        let context = Context::new_with_registry(&registry, false);
        context.set_allow_unregistered_dialects(true);
        context.load_all_available_dialects();
        context
    }

    // The mnemonics must match Enzyme's `Activity` enum (`enzyme/Enzyme/MLIR/Dialect/Attributes.td`)
    // exactly, since they're embedded in `#enzyme<activity ...>` attributes that only Enzyme's
    // parser can read back.
    #[test]
    fn activity_mnemonics() {
        assert_eq!(Activity::Active.mnemonic(), "enzyme_active");
        assert_eq!(Activity::Dup.mnemonic(), "enzyme_dup");
        assert_eq!(Activity::Const.mnemonic(), "enzyme_const");
        assert_eq!(Activity::DupNoNeed.mnemonic(), "enzyme_dupnoneed");
        assert_eq!(Activity::ActiveNoNeed.mnemonic(), "enzyme_activenoneed");
        assert_eq!(Activity::ConstNoNeed.mnemonic(), "enzyme_constnoneed");
    }

    #[test]
    fn compile_autodiff() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f64_type: Type = Type::parse(&context, "f64").unwrap();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[f64_type, f64_type], &[f64_type]).into(),
            ),
            {
                let block = Block::new(&[(f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().into();
                let rhs = block.argument(1).unwrap().into();

                let result = block
                    .append_operation(autodiff(
                        &context,
                        FlatSymbolRefAttribute::new(&context, "square"),
                        &[lhs, rhs],
                        &[f64_type],
                        activity_array_attribute(&context, &[Activity::Dup]),
                        activity_array_attribute(&context, &[Activity::ActiveNoNeed]),
                        1,
                        false,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[result], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    // `enzyme.batch` needs no `#enzyme<...>` attributes, so it can be built (in generic form)
    // even without the `enzyme` dialect registered.
    #[test]
    fn compile_batch() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let tensor_type: Type = Type::parse(&context, "tensor<4xf64>").unwrap();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[tensor_type], &[tensor_type]).into()),
            {
                let block = Block::new(&[(tensor_type, location)]);
                let operand = block.argument(0).unwrap().into();

                let result = block
                    .append_operation(batch(
                        &context,
                        FlatSymbolRefAttribute::new(&context, "square"),
                        &[operand],
                        &[tensor_type],
                        &[4],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[result], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_fwddiff() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f64_type: Type = Type::parse(&context, "f64").unwrap();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[f64_type, f64_type], &[f64_type]).into(),
            ),
            {
                let block = Block::new(&[(f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().into();
                let rhs = block.argument(1).unwrap().into();

                let result = block
                    .append_operation(fwddiff(
                        &context,
                        FlatSymbolRefAttribute::new(&context, "square"),
                        &[lhs, rhs],
                        &[f64_type],
                        activity_array_attribute(&context, &[Activity::Dup]),
                        activity_array_attribute(&context, &[Activity::Dup]),
                        1,
                        false,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[result], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_jacobian() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        let f64_type: Type = Type::parse(&context, "f64").unwrap();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[f64_type, f64_type], &[f64_type]).into(),
            ),
            {
                let block = Block::new(&[(f64_type, location), (f64_type, location)]);
                let lhs = block.argument(0).unwrap().into();
                let rhs = block.argument(1).unwrap().into();

                let result = block
                    .append_operation(jacobian(
                        &context,
                        FlatSymbolRefAttribute::new(&context, "square"),
                        &[lhs, rhs],
                        &[f64_type],
                        activity_array_attribute(&context, &[Activity::Active]),
                        activity_array_attribute(&context, &[Activity::Active]),
                        1,
                        false,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[result], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }
}
