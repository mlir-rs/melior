//! `index` dialect.

use super::arith::CmpiPredicate;
use crate::{
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, Attribute, Identifier, Location,
        Operation, Value,
    },
    Context,
};

// spell-checker: disable

/// Creates an `index.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    value: IntegerAttribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("index.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), value.into())])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Creates an `index.cmp` operation.
pub fn cmp<'c>(
    context: &'c Context,
    predicate: CmpiPredicate,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("index.cmp", location)
        .add_attributes(&[(
            Identifier::new(context, "pred"),
            Attribute::parse(
                context,
                match predicate {
                    CmpiPredicate::Eq => "#index<cmp_predicate eq>",
                    CmpiPredicate::Ne => "#index<cmp_predicate ne>",
                    CmpiPredicate::Slt => "#index<cmp_predicate slt>",
                    CmpiPredicate::Sle => "#index<cmp_predicate sle>",
                    CmpiPredicate::Sgt => "#index<cmp_predicate sgt>",
                    CmpiPredicate::Sge => "#index<cmp_predicate sge>",
                    CmpiPredicate::Ult => "#index<cmp_predicate ult>",
                    CmpiPredicate::Ule => "#index<cmp_predicate ule>",
                    CmpiPredicate::Ugt => "#index<cmp_predicate ugt>",
                    CmpiPredicate::Uge => "#index<cmp_predicate uge>",
                },
            )
            .unwrap(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

melior_macro::binary_operations!(
    index,
    [
        add, and, ceildivs, ceildivu, divs, divu, floordivs, maxs, maxu, mins, minu, mul, or, rems,
        remu, shl, shrs, shru, sub, xor,
    ]
);

melior_macro::typed_unary_operations!(index, [casts, castu]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute}, block::BlockLike, operation::OperationLike, r#type::{FunctionType, IntegerType}, Block, Location, Module, Region, RegionLike, Type
        },
        test::load_all_dialects,
        Context,
    };

    fn create_context() -> Context {
        let context = Context::new();
        load_all_dialects(&context);
        context
    }

    fn compile_operation<'c>(
        context: &'c Context,
        operation: impl Fn(&Block<'c>) -> Operation<'c>,
        function_type: FunctionType<'c>,
    ) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let block = Block::new(
            &(0..function_type.input_count())
                .map(|index| (function_type.input(index).unwrap(), location))
                .collect::<Vec<_>>(),
        );

        let operation = operation(&block);
        let name = operation.name();
        let name = name.as_string_ref().as_str().unwrap();

        block.append_operation(func::r#return(
            &[block.append_operation(operation).result(0).unwrap().into()],
            location,
        ));

        let region = Region::new();
        region.append_block(block);

        let function = func::func(
            context,
            StringAttribute::new(context, "foo"),
            TypeAttribute::new(function_type.into()),
            region,
            &[],
            Location::unknown(context),
        );

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_constant() {
        let context = create_context();
        let index_type = Type::index(&context);

        compile_operation(
            &context,
            |_| {
                constant(
                    &context,
                    IntegerAttribute::new(index_type, 42),
                    Location::unknown(&context),
                )
            },
            FunctionType::new(&context, &[index_type], &[index_type]),
        );
    }

    #[test]
    fn compile_cmp() {
        let context = create_context();
        let index_type = Type::index(&context);

        compile_operation(
            &context,
            |block| {
                cmp(
                    &context,
                    CmpiPredicate::Eq,
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                    Location::unknown(&context),
                )
            },
            FunctionType::new(
                &context,
                &[index_type, index_type],
                &[IntegerType::new(&context, 1).into()],
            ),
        );
    }

    mod typed_unary {
        use super::*;

        #[test]
        fn compile_casts() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    casts(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                FunctionType::new(
                    &context,
                    &[Type::index(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_castu() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    castu(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                FunctionType::new(
                    &context,
                    &[Type::index(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }
    }

    #[test]
    fn compile_add() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::index(&context);

        let function = {
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(add(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(
                    FunctionType::new(&context, &[integer_type, integer_type], &[integer_type])
                        .into(),
                ),
                region,
                &[],
                Location::unknown(&context),
            )
        };

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }
}
