//! `llvm` dialect.

use crate::{
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Identifier, Location, Operation, Region, Type, Value,
    },
    Context,
};
pub use alloca_options::*;
use attributes::{ICmpPredicate, Linkage};
use global::GlobalValue;
pub use load_store_options::*;

mod alloca_options;
pub mod attributes;
pub mod global;
mod load_store_options;
pub mod r#type;

// spell-checker: disable

/// Creates a `llvm.extractvalue` operation.
pub fn extract_value<'c>(
    context: &'c Context,
    container: Value<'c, '_>,
    position: DenseI64ArrayAttribute<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.extractvalue", location)
        .add_attributes(&[(Identifier::new(context, "position"), position.into())])
        .add_operands(&[container])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.getelementptr` operation.
pub fn get_element_ptr<'c>(
    context: &'c Context,
    ptr: Value<'c, '_>,
    indices: DenseI32ArrayAttribute<'c>,
    element_type: Type<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                indices.into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(element_type).into(),
            ),
        ])
        .add_operands(&[ptr])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.getelementptr` operation with dynamic indices.
pub fn get_element_ptr_dynamic<'c, const N: usize>(
    context: &'c Context,
    ptr: Value<'c, '_>,
    indices: &[Value<'c, '_>; N],
    element_type: Type<'c>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(context, &[i32::MIN; N]).into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(element_type).into(),
            ),
        ])
        .add_operands(&[ptr])
        .add_operands(indices)
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.insertvalue` operation.
pub fn insert_value<'c>(
    context: &'c Context,
    container: Value<'c, '_>,
    position: DenseI64ArrayAttribute<'c>,
    value: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.insertvalue", location)
        .add_attributes(&[(Identifier::new(context, "position"), position.into())])
        .add_operands(&[container, value])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.mlir.undef` operation.
pub fn undef<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.mlir.undef", location)
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.mlir.poison` operation.
pub fn poison<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.mlir.poison", location)
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a zero value.
pub fn zero<'c>(r#type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("llvm.mlir.zero", location)
        .add_results(&[r#type])
        .build()
        .expect("valid operation")
}

/// Creates a null pointer.
#[deprecated]
pub fn nullptr<'c>(ptr_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
    zero(ptr_type, location)
}

/// Creates a `llvm.unreachable` operation.
pub fn unreachable(location: Location) -> Operation {
    OperationBuilder::new("llvm.unreachable", location)
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.bitcast` operation.
pub fn bitcast<'c>(
    argument: Value<'c, '_>,
    result: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.bitcast", location)
        .add_operands(&[argument])
        .add_results(&[result])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.alloca` operation.
pub fn alloca<'c>(
    context: &'c Context,
    array_size: Value<'c, '_>,
    ptr_type: Type<'c>,
    location: Location<'c>,
    extra_options: AllocaOptions<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.alloca", location)
        .add_operands(&[array_size])
        .add_attributes(&extra_options.into_attributes(context))
        .add_results(&[ptr_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.store` operation.
pub fn store<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    addr: Value<'c, '_>,
    location: Location<'c>,
    extra_options: LoadStoreOptions<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.store", location)
        .add_operands(&[value, addr])
        .add_attributes(&extra_options.into_attributes(context))
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.load` operation.
pub fn load<'c>(
    context: &'c Context,
    addr: Value<'c, '_>,
    r#type: Type<'c>,
    location: Location<'c>,
    extra_options: LoadStoreOptions<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.load", location)
        .add_operands(&[addr])
        .add_attributes(&extra_options.into_attributes(context))
        .add_results(&[r#type])
        .build()
        .expect("valid operation")
}

/// Create a `llvm.func` operation.
pub fn func<'c>(
    context: &'c Context,
    name: StringAttribute<'c>,
    r#type: TypeAttribute<'c>,
    region: Region<'c>,
    attributes: &[(Identifier<'c>, Attribute<'c>)],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.func", location)
        .add_attributes(&[
            (Identifier::new(context, "sym_name"), name.into()),
            (Identifier::new(context, "function_type"), r#type.into()),
        ])
        .add_attributes(attributes)
        .add_regions([region])
        .build()
        .expect("valid operation")
}

// Creates a `llvm.return` operation.
pub fn r#return<'c>(value: Option<Value<'c, '_>>, location: Location<'c>) -> Operation<'c> {
    let mut builder = OperationBuilder::new("llvm.return", location);

    if let Some(value) = value {
        builder = builder.add_operands(&[value]);
    }

    builder.build().expect("valid operation")
}

/// Creates a `llvm.call_intrinsic` operation.
pub fn call_intrinsic<'c>(
    context: &'c Context,
    intrin: StringAttribute<'c>,
    args: &[Value<'c, '_>],
    results: &[Type<'c>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.call_intrinsic", location)
        .add_operands(args)
        .add_attributes(&[(Identifier::new(context, "intrin"), intrin.into())])
        .add_results(results)
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.ctlz` operation.
pub fn intr_ctlz<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    is_zero_poison: bool,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.ctlz", location)
        .add_attributes(&[(
            Identifier::new(context, "is_zero_poison"),
            IntegerAttribute::new(IntegerType::new(context, 1).into(), is_zero_poison.into())
                .into(),
        )])
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.ctlz` operation.
pub fn intr_cttz<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    is_zero_poison: bool,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.cttz", location)
        .add_attributes(&[(
            Identifier::new(context, "is_zero_poison"),
            IntegerAttribute::new(IntegerType::new(context, 1).into(), is_zero_poison.into())
                .into(),
        )])
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.ctlz` operation.
pub fn intr_ctpop<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.ctpop", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.bswap` operation.
pub fn intr_bswap<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.bswap", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.bitreverse` operation.
pub fn intr_bitreverse<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.bitreverse", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.intr.abs` operation.
pub fn intr_abs<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    is_int_min_poison: bool,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.intr.abs", location)
        .add_attributes(&[(
            Identifier::new(context, "is_int_min_poison"),
            IntegerAttribute::new(
                IntegerType::new(context, 1).into(),
                is_int_min_poison.into(),
            )
            .into(),
        )])
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates a `llvm.zext` operation.
pub fn zext<'c>(
    value: Value<'c, '_>,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("llvm.zext", location)
        .add_operands(&[value])
        .add_results(&[result_type])
        .build()
        .expect("valid operation")
}

/// Creates an `llvm.icmp` operation in Melior.
/// This is used to create integer AND pointer comparisons, unlike arith::icmp.
/// The predicate is an enum that specifies the type of comparison to be
/// performed. The result type is always a 1-bit integer (boolean).
/// The `lhs` and `rhs` operands are the values to be compared.
/// The `location` parameter specifies the location of the operation in the
/// source code.
pub fn icmp<'a>(
    context: &'a Context,
    predicate: ICmpPredicate,
    lhs: Value<'a, '_>,
    rhs: Value<'a, '_>,
    location: Location<'a>,
) -> Operation<'a> {
    OperationBuilder::new("llvm.icmp", location)
        .add_attributes(&[(
            Identifier::new(context, "predicate"),
            IntegerAttribute::new(IntegerType::new(context, 64).into(), predicate as i64).into(),
        )])
        .add_operands(&[lhs, rhs])
        .add_results(&[IntegerType::new(context, 1).into()])
        .build()
        .unwrap()
}

/// Creates an `llvm.mlir.addressof` operation to get a pointer to a global
/// variable.
pub fn addressof<'a>(
    context: &'a Context,
    global_name: &str,
    ptr_type: Type<'a>,
    location: Location<'a>,
) -> Operation<'a> {
    let location = Location::name(context, global_name, location);
    let global_name_attr = FlatSymbolRefAttribute::new(context, global_name);

    OperationBuilder::new("llvm.mlir.addressof", location)
        .add_attributes(&[(
            Identifier::new(context, "global_name"),
            global_name_attr.into(),
        )])
        .add_results(&[ptr_type])
        .build()
        .unwrap()
}

/// Creates an `llvm.mlir.global` operation in Melior.
pub fn global_variable<'a>(
    context: &'a Context,
    name: &str,
    value: GlobalValue<'a>,
    global_type: Type<'a>,
    options: global::GlobalVariableOptions,
    location: Location<'a>,
) -> Operation<'a> {
    let name_attr = StringAttribute::new(context, name);
    let type_attr = TypeAttribute::new(global_type);
    let addr_space_attr = IntegerAttribute::new(
        IntegerType::new(context, 32).into(),
        options.addr_space.unwrap_or(0).into(),
    );
    let alignment_attr = IntegerAttribute::new(
        IntegerType::new(context, 64).into(),
        options.alignment.unwrap_or(1),
    );

    let linkage_attr: Attribute<'_> =
        attributes::linkage(context, options.linkage.unwrap_or(Linkage::Internal));

    let mut op_builder = OperationBuilder::new("llvm.mlir.global", location).add_attributes(&[
        (Identifier::new(context, "sym_name"), name_attr.into()),
        (Identifier::new(context, "global_type"), type_attr.into()),
        (Identifier::new(context, "linkage"), linkage_attr),
        (
            Identifier::new(context, "addr_space"),
            addr_space_attr.into(),
        ),
        (Identifier::new(context, "alignment"), alignment_attr.into()),
    ]);

    match value {
        GlobalValue::Value(value) => {
            op_builder = op_builder
                .add_attributes(&[(Identifier::new(context, "value"), value)])
                .add_regions([Region::new()]);
        }
        GlobalValue::Constant(value) => {
            op_builder = op_builder
                .add_attributes(&[
                    (Identifier::new(context, "value"), value),
                    (
                        Identifier::new(context, "constant"),
                        Attribute::unit(context),
                    ),
                ])
                .add_regions([Region::new()]);
        }
        GlobalValue::Complex(region) => {
            op_builder = op_builder.add_regions([region]);
        }
    }

    op_builder.build().expect("valid operation")
}

/// Creates an `llvm.mlir.constant` operation in Melior.
///
/// Supports constant arrays and other complex types.
pub fn constant<'a>(
    context: &'a Context,
    value: Attribute<'a>,
    result_type: Type<'a>,
    location: Location<'a>,
) -> Operation<'a> {
    OperationBuilder::new("llvm.mlir.constant", location)
        .add_attributes(&[(Identifier::new(context, "value"), value)])
        .add_results(&[result_type])
        .build()
        .unwrap()
}

/// Creates an `llvm.call` operation in Melior.
/// Used for calling functions declared through LLVM dialect.
pub fn call<'a>(
    context: &'a Context,
    callee: &str,
    args: &[Value<'a, '_>],
    results: &[Type<'a>],
    location: Location<'a>,
) -> Operation<'a> {
    OperationBuilder::new("llvm.call", location)
        .add_attributes(&[(
            Identifier::new(context, "callee"),
            FlatSymbolRefAttribute::new(context, callee).into(),
        )])
        .add_operands(args)
        .add_results(results)
        .build()
        .unwrap()
}

/// Creates an `llvm.call` operation in Melior.
///
/// This can call any function from a pointer value.
pub fn indirect_call<'a>(
    callee: Value<'a, '_>,
    args: &[Value<'a, '_>],
    results: &[Type<'a>],
    location: Location<'a>,
) -> Operation<'a> {
    OperationBuilder::new("llvm.call", location)
        .add_operands(&[callee])
        .add_operands(args)
        .add_results(results)
        .build()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{
            arith, func,
            llvm::{
                attributes::{linkage, Linkage},
                r#type::function,
            },
        },
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::{FunctionType, IntegerType},
            Block, Module, Region, RegionLike,
        },
        pass::{self, PassManager},
        test::create_test_context,
    };
    use tests::r#type::pointer;

    fn convert_module<'c>(context: &'c Context, module: &mut Module<'c>) {
        let pass_manager = PassManager::new(context);

        pass_manager.add_pass(pass::conversion::create_func_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());
        pass_manager
            .nested_under("func.func")
            .add_pass(pass::conversion::create_index_to_llvm());
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());

        assert_eq!(pass_manager.run(module), Ok(()));
        assert!(module.as_operation().verify());
    }

    #[test]
    fn compile_extract_value() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(extract_value(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI64ArrayAttribute::new(&context, &[0]),
                    integer_type,
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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_get_element_ptr() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location)]);

                block.append_operation(get_element_ptr(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI32ArrayAttribute::new(&context, &[42]),
                    integer_type,
                    ptr_type,
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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_get_element_ptr_dynamic() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location)]);

                let index = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(integer_type, 42).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(get_element_ptr_dynamic(
                    &context,
                    block.argument(0).unwrap().into(),
                    &[index],
                    integer_type,
                    ptr_type,
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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_insert_value() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);
                let value = block
                    .append_operation(arith::constant(
                        &context,
                        IntegerAttribute::new(integer_type, 42).into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(insert_value(
                    &context,
                    block.argument(0).unwrap().into(),
                    DenseI64ArrayAttribute::new(&context, &[0]),
                    value,
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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_zero() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[integer_type]).into()),
            {
                let block = Block::new(&[]);

                let operation = block.append_operation(zero(integer_type, location));

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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_null_ptr() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[ptr_type]).into()),
            {
                let block = Block::new(&[]);

                let operation = block.append_operation(zero(ptr_type, location));

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

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_undefined() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let struct_type =
            r#type::r#struct(&context, &[IntegerType::new(&context, 64).into()], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(undef(struct_type, location));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_poison() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let struct_type =
            r#type::r#struct(&context, &[IntegerType::new(&context, 64).into()], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(poison(struct_type, location));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_alloca() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[integer_type], &[]).into()),
            {
                let block = Block::new(&[(integer_type, location)]);

                block.append_operation(alloca(
                    &context,
                    block.argument(0).unwrap().into(),
                    ptr_type,
                    location,
                    AllocaOptions::new().elem_type(Some(TypeAttribute::new(integer_type))),
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_store() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type, integer_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location), (integer_type, location)]);

                block.append_operation(store(
                    &context,
                    block.argument(1).unwrap().into(),
                    block.argument(0).unwrap().into(),
                    location,
                    Default::default(),
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_load() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location)]);

                block.append_operation(load(
                    &context,
                    block.argument(0).unwrap().into(),
                    integer_type,
                    location,
                    Default::default(),
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_store_extra() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[ptr_type, integer_type], &[]).into()),
            {
                let block = Block::new(&[(ptr_type, location), (integer_type, location)]);

                block.append_operation(store(
                    &context,
                    block.argument(1).unwrap().into(),
                    block.argument(0).unwrap().into(),
                    location,
                    LoadStoreOptions::new()
                        .align(Some(IntegerAttribute::new(integer_type, 4)))
                        .volatile(true)
                        .nontemporal(true),
                ));

                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_func() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func(
            &context,
            StringAttribute::new(&context, "printf"),
            TypeAttribute::new(function(integer_type, &[pointer(&context, 0)], true)),
            Region::new(),
            &[(
                Identifier::new(&context, "linkage"),
                linkage(&context, Linkage::External),
            )],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_return() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let struct_type = r#type::r#struct(&context, &[integer_type], false);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo_none"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(r#return(None, location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(FunctionType::new(&context, &[struct_type], &[struct_type]).into()),
            {
                let block = Block::new(&[(struct_type, location)]);

                block.append_operation(r#return(Some(block.argument(0).unwrap().into()), location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_ctlz() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_ctlz(
                        &context,
                        block.argument(0).unwrap().into(),
                        true,
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_cttz() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_cttz(
                        &context,
                        block.argument(0).unwrap().into(),
                        true,
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_ctpop() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_ctpop(
                        block.argument(0).unwrap().into(),
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_bswap() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_bswap(
                        block.argument(0).unwrap().into(),
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_bitreverse() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_bitreverse(
                        block.argument(0).unwrap().into(),
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_intr_abs() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(intr_abs(
                        &context,
                        block.argument(0).unwrap().into(),
                        true,
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_zext() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let integer_double_type = IntegerType::new(&context, 128).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_double_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(zext(
                        block.argument(0).unwrap().into(),
                        integer_double_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_icmp() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "foo"),
            TypeAttribute::new(
                FunctionType::new(
                    &context,
                    &[integer_type, integer_type],
                    &[IntegerType::new(&context, 1).into()],
                )
                .into(),
            ),
            {
                let block = Block::new(&[(integer_type, location), (integer_type, location)]);

                let res = block
                    .append_operation(icmp(
                        &context,
                        ICmpPredicate::Eq,
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_addressof() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = r#type::pointer(&context, 0);

        // First create a global variable
        module.body().append_operation(global_variable(
            &context,
            "my_global",
            GlobalValue::Constant(IntegerAttribute::new(integer_type, 42).into()),
            integer_type,
            global::GlobalVariableOptions::default(),
            location,
        ));

        // Then a function that gets the address of it
        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "get_global_addr"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[ptr_type]).into()),
            {
                let block = Block::new(&[]);

                let res = block
                    .append_operation(addressof(&context, "my_global", ptr_type, location))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_global_variable() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        // Global variable with a constant value
        module.body().append_operation(global_variable(
            &context,
            "const_global",
            GlobalValue::Constant(IntegerAttribute::new(integer_type, 42).into()),
            integer_type,
            global::GlobalVariableOptions {
                linkage: Some(Linkage::Internal),
                alignment: Some(4),
                addr_space: Some(0),
            },
            location,
        ));

        // Global variable with a value (not constant)
        module.body().append_operation(global_variable(
            &context,
            "mutable_global",
            GlobalValue::Value(IntegerAttribute::new(integer_type, 123).into()),
            integer_type,
            global::GlobalVariableOptions {
                linkage: Some(Linkage::External),
                alignment: Some(8),
                addr_space: Some(0),
            },
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_constant() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "get_constant"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[integer_type]).into()),
            {
                let block = Block::new(&[]);

                let res = block
                    .append_operation(constant(
                        &context,
                        IntegerAttribute::new(integer_type, 42).into(),
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_call() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        // First define a function
        module.body().append_operation(func(
            &context,
            StringAttribute::new(&context, "add_one"),
            TypeAttribute::new(function(integer_type, &[integer_type], false)),
            {
                let block = Block::new(&[(integer_type, location)]);

                let one = block
                    .append_operation(constant(
                        &context,
                        IntegerAttribute::new(integer_type, 1).into(),
                        integer_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                let result = block
                    .append_operation(arith::addi(
                        block.argument(0).unwrap().into(),
                        one,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(r#return(Some(result), location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        // Then a function that calls it
        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "caller"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(call(
                        &context,
                        "add_one",
                        &[block.argument(0).unwrap().into()],
                        &[integer_type],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_indirect_call() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();
        let fn_type = function(integer_type, &[integer_type], false);
        let fn_ptr_type = pointer(&context, 0);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "call_fn_ptr"),
            TypeAttribute::new(
                FunctionType::new(&context, &[fn_ptr_type, integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(fn_ptr_type, location), (integer_type, location)]);

                let res = block
                    .append_operation(indirect_call(
                        block.argument(0).unwrap().into(),
                        &[block.argument(1).unwrap().into()],
                        &[integer_type],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_call_intrinsic() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let integer_type = IntegerType::new(&context, 32).into();

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "use_intrinsic"),
            TypeAttribute::new(
                FunctionType::new(&context, &[integer_type], &[integer_type]).into(),
            ),
            {
                let block = Block::new(&[(integer_type, location)]);

                let res = block
                    .append_operation(call_intrinsic(
                        &context,
                        StringAttribute::new(&context, "llvm.bitreverse.i32"),
                        &[block.argument(0).unwrap().into()],
                        &[integer_type],
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_bitcast() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let i32_type = IntegerType::new(&context, 32).into();
        let float_type = Type::float32(&context);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "int_bits_to_float"),
            TypeAttribute::new(FunctionType::new(&context, &[i32_type], &[float_type]).into()),
            {
                let block = Block::new(&[(i32_type, location)]);

                let res = block
                    .append_operation(bitcast(
                        block.argument(0).unwrap().into(),
                        float_type,
                        location,
                    ))
                    .result(0)
                    .unwrap()
                    .into();

                block.append_operation(func::r#return(&[res], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_unreachable() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "func_with_unreachable"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(unreachable(location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));

        convert_module(&context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }
}
