//! `arith` dialect.

use crate::{
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Attribute,
        Identifier, Location, Operation, Value, ValueLike,
    },
    Context,
};

// spell-checker: disable

/// Creates an `arith.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    value: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("arith.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), value)])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// `arith.cmpf` predicate
pub enum CmpfPredicate {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

/// Creates an `arith.cmpf` operation.
pub fn cmpf<'c>(
    context: &'c Context,
    predicate: CmpfPredicate,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    cmp(context, "arith.cmpf", predicate as i64, lhs, rhs, location)
}

/// `arith.cmpi` predicate
pub enum CmpiPredicate {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

/// Creates an `arith.cmpi` operation.
pub fn cmpi<'c>(
    context: &'c Context,
    predicate: CmpiPredicate,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    cmp(context, "arith.cmpi", predicate as i64, lhs, rhs, location)
}

fn cmp<'c>(
    context: &'c Context,
    name: &str,
    predicate: i64,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new(name, location)
        .add_attributes(&[(
            Identifier::new(context, "predicate"),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), predicate).into(),
        )])
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Creates an `arith.select` operation.
pub fn select<'c>(
    condition: Value<'c, '_>,
    true_value: Value<'c, '_>,
    false_value: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("arith.select", location)
        .add_operands(&[condition, true_value, false_value])
        .add_results(&[true_value.r#type()])
        .build()
        .expect("valid operation")
}

melior_macro::binary_operations!(
    arith,
    [
        addf,
        addi,
        addui_extended,
        andi,
        ceildivsi,
        ceildivui,
        divf,
        divsi,
        divui,
        floordivsi,
        maxnumf,
        maxsi,
        maxui,
        minimumf,
        minsi,
        minui,
        mulf,
        muli,
        mulsi_extended,
        mului_extended,
        ori,
        remf,
        remsi,
        remui,
        shli,
        shrsi,
        shrui,
        subf,
        subi,
        xori,
    ]
);

melior_macro::unary_operations!(arith, [negf, truncf]);

melior_macro::typed_unary_operations!(
    arith,
    [
        bitcast,
        extf,
        extsi,
        extui,
        fptosi,
        fptoui,
        index_cast,
        index_castui,
        sitofp,
        trunci,
        uitofp
    ]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            block::BlockLike,
            r#type::FunctionType,
            Attribute, Block, Location, Module, Region, RegionLike, Type,
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
        block_argument_types: &[Type<'c>],
        function_type: FunctionType<'c>,
    ) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let block = Block::new(
            &block_argument_types
                .iter()
                .map(|&r#type| (r#type, location))
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
        let integer_type = IntegerType::new(&context, 64).into();

        compile_operation(
            &context,
            |_| {
                constant(
                    &context,
                    Attribute::parse(&context, "42 : i64").unwrap(),
                    Location::unknown(&context),
                )
            },
            &[integer_type],
            FunctionType::new(&context, &[integer_type], &[integer_type]),
        );
    }

    #[test]
    fn compile_negf() {
        let context = create_context();
        let f64_type = Type::float64(&context);

        compile_operation(
            &context,
            |block| {
                negf(
                    block.argument(0).unwrap().into(),
                    Location::unknown(&context),
                )
            },
            &[Type::float64(&context)],
            FunctionType::new(&context, &[f64_type], &[f64_type]),
        );
    }

    mod cmp {
        use super::*;

        #[test]
        fn compile_cmpf() {
            let context = create_context();
            let float_type = Type::float64(&context);

            compile_operation(
                &context,
                |block| {
                    cmpf(
                        &context,
                        CmpfPredicate::Oeq,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        Location::unknown(&context),
                    )
                },
                &[float_type, float_type],
                FunctionType::new(
                    &context,
                    &[float_type, float_type],
                    &[IntegerType::new(&context, 1).into()],
                ),
            );
        }

        #[test]
        fn compile_cmpi() {
            let context = create_context();
            let integer_type = IntegerType::new(&context, 64).into();

            compile_operation(
                &context,
                |block| {
                    cmpi(
                        &context,
                        CmpiPredicate::Eq,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        Location::unknown(&context),
                    )
                },
                &[integer_type, integer_type],
                FunctionType::new(
                    &context,
                    &[integer_type, integer_type],
                    &[IntegerType::new(&context, 1).into()],
                ),
            );
        }
    }

    mod typed_unary {
        use super::*;

        #[test]
        fn compile_bitcast() {
            let context = create_context();
            let integer_type = IntegerType::new(&context, 64).into();
            let float_type = Type::float64(&context);

            compile_operation(
                &context,
                |block| {
                    bitcast(
                        block.argument(0).unwrap().into(),
                        float_type,
                        Location::unknown(&context),
                    )
                },
                &[integer_type],
                FunctionType::new(&context, &[integer_type], &[float_type]),
            );
        }

        #[test]
        fn compile_extf() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extf(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                FunctionType::new(
                    &context,
                    &[Type::float32(&context)],
                    &[Type::float64(&context)],
                ),
            );
        }

        #[test]
        fn compile_extsi() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extsi(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[IntegerType::new(&context, 32).into()],
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_extui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    extui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[IntegerType::new(&context, 32).into()],
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_fptosi() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    fptosi(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                FunctionType::new(
                    &context,
                    &[Type::float32(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_fptoui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    fptoui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[Type::float32(&context)],
                FunctionType::new(
                    &context,
                    &[Type::float32(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_index_cast() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    index_cast(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[Type::index(&context)],
                FunctionType::new(
                    &context,
                    &[Type::index(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_index_castui() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    index_castui(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 64).into(),
                        Location::unknown(&context),
                    )
                },
                &[Type::index(&context)],
                FunctionType::new(
                    &context,
                    &[Type::index(&context)],
                    &[IntegerType::new(&context, 64).into()],
                ),
            );
        }

        #[test]
        fn compile_sitofp() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    sitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[IntegerType::new(&context, 32).into()],
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[Type::float64(&context)],
                ),
            );
        }

        #[test]
        fn compile_trunci() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    trunci(
                        block.argument(0).unwrap().into(),
                        IntegerType::new(&context, 32).into(),
                        Location::unknown(&context),
                    )
                },
                &[IntegerType::new(&context, 64).into()],
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 64).into()],
                    &[IntegerType::new(&context, 32).into()],
                ),
            );
        }

        #[test]
        fn compile_uitofp() {
            let context = create_context();

            compile_operation(
                &context,
                |block| {
                    uitofp(
                        block.argument(0).unwrap().into(),
                        Type::float64(&context),
                        Location::unknown(&context),
                    )
                },
                &[IntegerType::new(&context, 32).into()],
                FunctionType::new(
                    &context,
                    &[IntegerType::new(&context, 32).into()],
                    &[Type::float64(&context)],
                ),
            );
        }
    }

    #[test]
    fn compile_addi() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = IntegerType::new(&context, 64).into();

        let function = {
            let block = Block::new(&[(integer_type, location), (integer_type, location)]);

            let sum = block.append_operation(addi(
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

    #[test]
    fn compile_select() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = IntegerType::new(&context, 64).into();
        let bool_type = IntegerType::new(&context, 1).into();

        let function = {
            let block = Block::new(&[
                (bool_type, location),
                (integer_type, location),
                (integer_type, location),
            ]);

            let val = block.append_operation(select(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                block.argument(2).unwrap().into(),
                location,
            ));

            block.append_operation(func::r#return(&[val.result(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(
                    FunctionType::new(
                        &context,
                        &[bool_type, integer_type, integer_type],
                        &[integer_type],
                    )
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
