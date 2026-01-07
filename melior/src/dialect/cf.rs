//! `cf` dialect.

use crate::{
    Context, Error,
    ir::{
        Block, Identifier, Location, Operation, Type, Value,
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, IntegerAttribute, StringAttribute,
        },
        block::BlockLike,
        operation::OperationBuilder,
        r#type::RankedTensorType,
    },
};

/// Creates a `cf.assert` operation.
pub fn assert<'c>(
    context: &'c Context,
    argument: Value<'c, '_>,
    message: &str,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.assert", location)
        .add_attributes(&[(
            Identifier::new(context, "msg"),
            StringAttribute::new(context, message).into(),
        )])
        .add_operands(&[argument])
        .build()
        .expect("valid operation")
}

/// Creates a `cf.br` operation.
pub fn br<'c>(
    successor: &Block<'c>,
    destination_operands: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.br", location)
        .add_operands(destination_operands)
        .add_successors(&[successor])
        .build()
        .expect("valid operation")
}

/// Creates a `cf.cond_br` operation.
pub fn cond_br<'c>(
    context: &'c Context,
    condition: Value<'c, '_>,
    true_successor: &Block<'c>,
    false_successor: &Block<'c>,
    true_successor_operands: &[Value<'c, '_>],
    false_successor_operands: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("cf.cond_br", location)
        .add_attributes(&[(
            Identifier::new(context, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(
                context,
                &[
                    1,
                    true_successor.argument_count() as i32,
                    false_successor.argument_count() as i32,
                ],
            )
            .into(),
        )])
        .add_operands(
            &[condition]
                .into_iter()
                .chain(true_successor_operands.iter().copied())
                .chain(false_successor_operands.iter().copied())
                .collect::<Vec<_>>(),
        )
        .add_successors(&[true_successor, false_successor])
        .build()
        .expect("valid operation")
}

/// Creates a `cf.switch` operation.
pub fn switch<'c>(
    context: &'c Context,
    case_values: &[i64],
    flag: Value<'c, '_>,
    flag_type: Type<'c>,
    default_destination: (&Block<'c>, &[Value<'c, '_>]),
    case_destinations: &[(&Block<'c>, &[Value<'c, '_>])],
    location: Location<'c>,
) -> Result<Operation<'c>, Error> {
    let (destinations, operands): (Vec<_>, Vec<_>) = [default_destination]
        .into_iter()
        .chain(case_destinations.iter().copied())
        .unzip();

    Ok(OperationBuilder::new("cf.switch", location)
        .add_attributes(&[
            (
                Identifier::new(context, "case_values"),
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[case_values.len() as u64], flag_type, None).into(),
                    &case_values
                        .iter()
                        .map(|value| IntegerAttribute::new(flag_type, *value).into())
                        .collect::<Vec<_>>(),
                )?
                .into(),
            ),
            (
                Identifier::new(context, "case_operand_segments"),
                DenseI32ArrayAttribute::new(
                    context,
                    &case_destinations
                        .iter()
                        .map(|(_, operands)| operands.len() as i32)
                        .collect::<Vec<_>>(),
                )
                .into(),
            ),
            (
                Identifier::new(context, "operand_segment_sizes"),
                DenseI32ArrayAttribute::new(
                    context,
                    &[
                        1,
                        default_destination.1.len() as i32,
                        case_destinations
                            .iter()
                            .map(|(_, operands)| operands.len() as i32)
                            .sum(),
                    ],
                )
                .into(),
            ),
        ])
        .add_operands(
            &[flag]
                .into_iter()
                .chain(operands.into_iter().flatten().copied())
                .collect::<Vec<_>>(),
        )
        .add_successors(&destinations)
        .build()
        .expect("valid operation"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        dialect::{
            arith::{self, CmpiPredicate},
            func, index,
        },
        ir::{
            Block, Module, Region, RegionLike,
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            operation::OperationLike,
            r#type::{FunctionType, IntegerType, Type},
        },
        test::load_all_dialects,
    };

    #[test]
    fn compile_assert() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let bool_type: Type = IntegerType::new(&context, 1).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let operand = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(bool_type, 1).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(assert(&context, operand, "assert message", location));

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
    fn compile_br() {
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
                let dest_block = Block::new(&[(index_type, location)]);
                let operand = block
                    .append_operation(index::constant(
                        &context,
                        IntegerAttribute::new(index_type, 1),
                        location,
                    ))
                    .result(0)
                    .unwrap();

                block.append_operation(br(&dest_block, &[operand.into()], location));

                dest_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(dest_block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_cond_br() {
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
                let true_block = Block::new(&[(index_type, location)]);
                let false_block = Block::new(&[(index_type, location)]);

                let operand = block
                    .append_operation(index::constant(
                        &context,
                        IntegerAttribute::new(index_type, 1),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                let condition = block
                    .append_operation(index::cmp(
                        &context,
                        CmpiPredicate::Eq,
                        operand,
                        operand,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(cond_br(
                    &context,
                    condition,
                    &true_block,
                    &false_block,
                    &[operand],
                    &[operand],
                    location,
                ));

                true_block.append_operation(func::r#return(&[], location));
                false_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(true_block);
                region.append_block(false_block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_switch() {
        let context = Context::new();
        load_all_dialects(&context);

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let i32_type: Type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                let default_block = Block::new(&[(i32_type, location)]);
                let first_block = Block::new(&[(i32_type, location)]);
                let second_block = Block::new(&[(i32_type, location)]);

                let operand = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(i32_type, 1).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(
                    switch(
                        &context,
                        &[0, 1],
                        operand,
                        i32_type,
                        (&default_block, &[operand]),
                        &[(&first_block, &[operand]), (&second_block, &[operand])],
                        location,
                    )
                    .unwrap(),
                );

                default_block.append_operation(func::r#return(&[], location));
                first_block.append_operation(func::r#return(&[], location));
                second_block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region.append_block(default_block);
                region.append_block(first_block);
                region.append_block(second_block);
                region
            },
            &[],
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }
}
