//! `scf` dialect.

use crate::{
    ir::{
        attribute::DenseI64ArrayAttribute, operation::OperationBuilder, Identifier, Location,
        Operation, Region, Type, Value,
    },
    Context,
};

/// Creates a `scf.condition` operation.
pub fn condition<'c>(
    condition: Value<'c, '_>,
    values: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.condition", location)
        .add_operands(&[condition])
        .add_operands(values)
        .build()
        .expect("valid operation")
}

/// Creates a `scf.execute_region` operation.
pub fn execute_region<'c>(
    result_types: &[Type<'c>],
    region: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.execute_region", location)
        .add_results(result_types)
        .add_regions([region])
        .build()
        .expect("valid operation")
}

/// Creates a `scf.for` operation.
pub fn r#for<'c>(
    start: Value<'c, '_>,
    end: Value<'c, '_>,
    step: Value<'c, '_>,
    region: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.for", location)
        .add_operands(&[start, end, step])
        .add_regions([region])
        .build()
        .expect("valid operation")
}

/// Creates a `scf.if` operation.
pub fn r#if<'c>(
    condition: Value<'c, '_>,
    result_types: &[Type<'c>],
    then_region: Region<'c>,
    else_region: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.if", location)
        .add_operands(&[condition])
        .add_results(result_types)
        .add_regions([then_region, else_region])
        .build()
        .expect("valid operation")
}

/// Creates a `scf.index_switch` operation.
pub fn index_switch<'c>(
    context: &'c Context,
    condition: Value<'c, '_>,
    result_types: &[Type<'c>],
    cases: DenseI64ArrayAttribute<'c>,
    regions: Vec<Region<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.index_switch", location)
        .add_operands(&[condition])
        .add_results(result_types)
        .add_attributes(&[(Identifier::new(context, "cases"), cases.into())])
        .add_regions_vec(regions)
        .build()
        .expect("valid operation")
}

/// Creates a `scf.while` operation.
pub fn r#while<'c>(
    initial_values: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    before_region: Region<'c>,
    after_region: Region<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("scf.while", location)
        .add_operands(initial_values)
        .add_results(result_types)
        .add_regions([before_region, after_region])
        .build()
        .expect("valid operation")
}

/// Creates a `scf.yield` operation.
pub fn r#yield<'c>(values: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("scf.yield", location)
        .add_operands(values)
        .build()
        .expect("valid operation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{arith, func},
        ir::{
            attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::{FunctionType, IntegerType, Type},
            Attribute, Block, Module, RegionLike,
        },
        test::load_all_dialects,
        Context,
    };

    #[test]
    fn compile_execute_region() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(execute_region(
                    &[index_type],
                    {
                        let block = Block::new(&[]);

                        let value = block.append_operation(arith::constant(
                            &context,
                            IntegerAttribute::new(index_type, 0).into(),
                            location,
                        ));

                        block.append_operation(r#yield(
                            &[value.result(0).unwrap().into()],
                            location,
                        ));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

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
    fn compile_for() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                let start = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "0 : index").unwrap(),
                    location,
                ));

                let end = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "8 : index").unwrap(),
                    location,
                ));

                let step = block.append_operation(arith::constant(
                    &context,
                    Attribute::parse(&context, "1 : index").unwrap(),
                    location,
                ));

                block.append_operation(r#for(
                    start.result(0).unwrap().into(),
                    end.result(0).unwrap().into(),
                    step.result(0).unwrap().into(),
                    {
                        let block = Block::new(&[(Type::index(&context), location)]);
                        block.append_operation(r#yield(&[], location));

                        let region = Region::new();
                        region.append_block(block);
                        region
                    },
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

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

    mod r#if {
        use super::*;

        #[test]
        fn compile() {
            let context = Context::new();
            load_all_dialects(&context);

            let location = Location::unknown(&context);
            let module = Module::new(location);
            let index_type = Type::index(&context);

            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[index_type]).into()),
                {
                    let block = Block::new(&[]);

                    let condition = block.append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(IntegerType::new(&context, 1).into(), 0).into(),
                        location,
                    ));

                    let result = block.append_operation(r#if(
                        condition.result(0).unwrap().into(),
                        &[index_type],
                        {
                            let block = Block::new(&[]);

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(index_type, 42).into(),
                                location,
                            ));

                            block.append_operation(r#yield(
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(index_type, 13).into(),
                                location,
                            ));

                            block.append_operation(r#yield(
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        location,
                    ));

                    block.append_operation(func::r#return(
                        &[result.result(0).unwrap().into()],
                        location,
                    ));

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
        fn compile_one_sided() {
            let context = Context::new();
            load_all_dialects(&context);

            let location = Location::unknown(&context);
            let module = Module::new(location);

            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                {
                    let block = Block::new(&[]);

                    let condition = block.append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(IntegerType::new(&context, 1).into(), 0).into(),
                        location,
                    ));

                    block.append_operation(r#if(
                        condition.result(0).unwrap().into(),
                        &[],
                        {
                            let block = Block::new(&[]);

                            block.append_operation(r#yield(&[], location));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        Region::new(),
                        location,
                    ));

                    block.append_operation(func::r#return(&[], location));

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

    #[test]
    fn compile_index_switch() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                let condition = block.append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(Type::index(&context), 0).into(),
                    location,
                ));

                block.append_operation(index_switch(
                    &context,
                    condition.result(0).unwrap().into(),
                    &[],
                    DenseI64ArrayAttribute::new(&context, &[0, 1]),
                    vec![
                        {
                            let block = Block::new(&[]);

                            block.append_operation(r#yield(&[], location));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);

                            block.append_operation(r#yield(&[], location));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[]);

                            block.append_operation(r#yield(&[], location));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                    ],
                    location,
                ));

                block.append_operation(func::r#return(&[], location));

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

    mod r#while {
        use super::*;

        #[test]
        fn compile() {
            let context = Context::new();
            load_all_dialects(&context);

            let location = Location::unknown(&context);
            let module = Module::new(location);
            let index_type = Type::index(&context);

            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                {
                    let block = Block::new(&[]);

                    let initial = block.append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(index_type, 0).into(),
                        location,
                    ));

                    block.append_operation(r#while(
                        &[initial.result(0).unwrap().into()],
                        &[index_type],
                        {
                            let block = Block::new(&[(index_type, location)]);

                            let condition = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(IntegerType::new(&context, 1).into(), 0)
                                    .into(),
                                location,
                            ));

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(Type::index(&context), 42).into(),
                                location,
                            ));

                            block.append_operation(super::condition(
                                condition.result(0).unwrap().into(),
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[(index_type, location)]);

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(index_type, 42).into(),
                                location,
                            ));

                            block.append_operation(r#yield(
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        location,
                    ));

                    block.append_operation(func::r#return(&[], location));

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
        fn compile_with_different_argument_and_result_types() {
            let context = Context::new();
            load_all_dialects(&context);

            let location = Location::unknown(&context);
            let module = Module::new(location);
            let index_type = Type::index(&context);
            let float_type = Type::float64(&context);

            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                {
                    let block = Block::new(&[]);

                    let initial = block.append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(index_type, 0).into(),
                        location,
                    ));

                    block.append_operation(r#while(
                        &[initial.result(0).unwrap().into()],
                        &[float_type],
                        {
                            let block = Block::new(&[(index_type, location)]);

                            let condition = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(IntegerType::new(&context, 1).into(), 0)
                                    .into(),
                                location,
                            ));

                            let result = block.append_operation(arith::constant(
                                &context,
                                FloatAttribute::new(&context, float_type, 42.0).into(),
                                location,
                            ));

                            block.append_operation(super::condition(
                                condition.result(0).unwrap().into(),
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block = Block::new(&[(float_type, location)]);

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(Type::index(&context), 42).into(),
                                location,
                            ));

                            block.append_operation(r#yield(
                                &[result.result(0).unwrap().into()],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        location,
                    ));

                    block.append_operation(func::r#return(&[], location));

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
        fn compile_with_multiple_arguments_and_results() {
            let context = Context::new();
            load_all_dialects(&context);

            let location = Location::unknown(&context);
            let module = Module::new(location);
            let index_type = Type::index(&context);

            module.body().append_operation(func::func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
                {
                    let block = Block::new(&[]);

                    let initial = block.append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(index_type, 0).into(),
                        location,
                    ));

                    block.append_operation(r#while(
                        &[
                            initial.result(0).unwrap().into(),
                            initial.result(0).unwrap().into(),
                        ],
                        &[index_type, index_type],
                        {
                            let block =
                                Block::new(&[(index_type, location), (index_type, location)]);

                            let condition = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(IntegerType::new(&context, 1).into(), 0)
                                    .into(),
                                location,
                            ));

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(Type::index(&context), 42).into(),
                                location,
                            ));

                            block.append_operation(super::condition(
                                condition.result(0).unwrap().into(),
                                &[
                                    result.result(0).unwrap().into(),
                                    result.result(0).unwrap().into(),
                                ],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        {
                            let block =
                                Block::new(&[(index_type, location), (index_type, location)]);

                            let result = block.append_operation(arith::constant(
                                &context,
                                IntegerAttribute::new(index_type, 42).into(),
                                location,
                            ));

                            block.append_operation(r#yield(
                                &[
                                    result.result(0).unwrap().into(),
                                    result.result(0).unwrap().into(),
                                ],
                                location,
                            ));

                            let region = Region::new();
                            region.append_block(block);
                            region
                        },
                        location,
                    ));

                    block.append_operation(func::r#return(&[], location));

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
}
