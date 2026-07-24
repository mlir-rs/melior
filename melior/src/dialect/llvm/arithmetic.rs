//! Arithmetic operations for the `llvm` dialect.

use crate::ir::{Location, Operation, Value, operation::OperationBuilder};

/// Creates an `llvm.add` operation.
pub fn add<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.add", lhs, rhs, location)
}

/// Creates an `llvm.sub` operation.
pub fn sub<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.sub", lhs, rhs, location)
}

/// Creates an `llvm.mul` operation.
pub fn mul<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.mul", lhs, rhs, location)
}

/// Creates an `llvm.and` operation.
pub fn and<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.and", lhs, rhs, location)
}

/// Creates an `llvm.or` operation.
pub fn or<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.or", lhs, rhs, location)
}

/// Creates an `llvm.xor` operation.
pub fn xor<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    binary_operation("llvm.xor", lhs, rhs, location)
}

fn binary_operation<'c>(
    name: &str,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new(name, location)
        .add_operands(&[lhs, rhs])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{
            Block, BlockLike, Module, Region, RegionLike,
            attribute::{StringAttribute, TypeAttribute},
            operation::OperationLike,
            r#type::{FunctionType, IntegerType},
        },
        test::create_test_context,
    };

    macro_rules! compile_binary_operation {
        ($test_name:ident, $builder:ident) => {
            #[test]
            fn $test_name() {
                let context = create_test_context();

                let location = Location::unknown(&context);
                let module = Module::new(location);
                let integer_type = IntegerType::new(&context, 64).into();

                module.body().append_operation(func::func(
                    &context,
                    StringAttribute::new(&context, "foo"),
                    TypeAttribute::new(
                        FunctionType::new(&context, &[integer_type, integer_type], &[integer_type])
                            .into(),
                    ),
                    {
                        let block =
                            Block::new(&[(integer_type, location), (integer_type, location)]);

                        let operation = block.append_operation($builder(
                            block.argument(0).unwrap().into(),
                            block.argument(1).unwrap().into(),
                            location,
                        ));

                        block.append_operation(func::r#return(
                            &[operation.result(0).unwrap().into()],
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
        };
    }

    compile_binary_operation!(compile_add, add);
    compile_binary_operation!(compile_sub, sub);
    compile_binary_operation!(compile_mul, mul);
    compile_binary_operation!(compile_and, and);
    compile_binary_operation!(compile_or, or);
    compile_binary_operation!(compile_xor, xor);
}
