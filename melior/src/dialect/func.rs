//! `func` dialect.

use crate::{
    Context,
    ir::{
        Attribute, Identifier, Location, Operation, Region, Type, Value,
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::FunctionType,
    },
};

/// Create a `func.call` operation.
pub fn call<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    arguments: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.call", location)
        .add_attributes(&[(Identifier::new(context, "callee"), function.into())])
        .add_operands(arguments)
        .add_results(result_types)
        .build()
        .expect("valid operation")
}

/// Create a `func.call_indirect` operation.
pub fn call_indirect<'c>(
    function: Value<'c, '_>,
    arguments: &[Value<'c, '_>],
    result_types: &[Type<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.call_indirect", location)
        .add_operands(&[function])
        .add_operands(arguments)
        .add_results(result_types)
        .build()
        .expect("valid operation")
}

/// Create a `func.constant` operation.
pub fn constant<'c>(
    context: &'c Context,
    function: FlatSymbolRefAttribute<'c>,
    r#type: FunctionType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), function.into())])
        .add_results(&[r#type.into()])
        .build()
        .expect("valid operation")
}

/// Create a `func.func` operation.
pub fn func<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    r#type: TypeAttribute<'c>,
    region: Region<'c>,
    attributes: &[(Identifier<'c>, Attribute<'c>)],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("func.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), name.into()),
            (Identifier::new(context, "function_type"), r#type.into()),
        ])
        .add_attributes(attributes)
        .add_regions([region])
        .build()
        .expect("valid operation")
}

/// Create a `func.return` operation.
pub fn r#return<'c>(operands: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("func.return", location)
        .add_operands(operands)
        .build()
        .expect("valid operation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Block, Module, RegionLike, Type, block::BlockLike, operation::OperationLike},
        test::create_test_context,
    };

    #[test]
    fn compile_call() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);
        let function_type = FunctionType::new(&context, &[index_type], &[index_type]);

        let function = func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(function_type.into()),
            {
                let block = Block::new(&[(index_type, location)]);

                let value = block
                    .append_operation(call(
                        &context,
                        FlatSymbolRefAttribute::new(&context, "foo"),
                        &[block.argument(0).unwrap().into()],
                        &[index_type],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(r#return(&[value], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            Location::unknown(&context),
        );

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_call_indirect() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);
        let index_type = Type::index(&context);
        let function_type = FunctionType::new(&context, &[index_type], &[index_type]);

        let function = func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(function_type.into()),
            {
                let block = Block::new(&[(index_type, location)]);

                let function = block.append_operation(constant(
                    &context,
                    FlatSymbolRefAttribute::new(&context, "foo"),
                    function_type,
                    location,
                ));
                let value = block
                    .append_operation(call_indirect(
                        function.result(0).unwrap().into(),
                        &[block.argument(0).unwrap().into()],
                        &[index_type],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();
                block.append_operation(r#return(&[value], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            Location::unknown(&context),
        );

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_function() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::index(&context);

        let function = {
            let block = Block::new(&[(integer_type, location)]);

            block.append_operation(r#return(&[block.argument(0).unwrap().into()], location));

            let region = Region::new();
            region.append_block(block);

            func(
                &context,
                StringAttribute::new(&context, "foo"),
                TypeAttribute::new(
                    FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
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
    fn compile_external_function() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let module = Module::new(location);

        let integer_type = Type::index(&context);

        let function = func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            Region::new(),
            &[(
                Identifier::new(&context, "sym_visibility"),
                StringAttribute::new(&context, "private").into(),
            )],
            Location::unknown(&context),
        );

        module.body().append_operation(function);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }
}
