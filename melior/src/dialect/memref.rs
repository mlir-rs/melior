//! `memref` dialect.

use crate::{
    ir::{
        attribute::{
            DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::MemRefType,
        Attribute, Identifier, Location, Operation, Value,
    },
    Context,
};

/// Create a `memref.alloc` operation.
pub fn alloc<'c>(
    context: &'c Context,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value<'c, '_>],
    symbols: &[Value<'c, '_>],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    allocate(
        context,
        "memref.alloc",
        r#type,
        dynamic_sizes,
        symbols,
        alignment,
        location,
    )
}

/// Create a `memref.alloca` operation.
pub fn alloca<'c>(
    context: &'c Context,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value<'c, '_>],
    symbols: &[Value<'c, '_>],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    allocate(
        context,
        "memref.alloca",
        r#type,
        dynamic_sizes,
        symbols,
        alignment,
        location,
    )
}

fn allocate<'c>(
    context: &'c Context,
    name: &str,
    r#type: MemRefType<'c>,
    dynamic_sizes: &[Value<'c, '_>],
    symbols: &[Value<'c, '_>],
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new(name, location);

    builder = builder.add_attributes(&[(
        Identifier::new(context, "operand_segment_sizes"),
        DenseI32ArrayAttribute::new(context, &[dynamic_sizes.len() as i32, symbols.len() as i32])
            .into(),
    )]);
    builder = builder.add_operands(dynamic_sizes).add_operands(symbols);

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder
        .add_results(&[r#type.into()])
        .build()
        .expect("valid operation")
}

/// Create a `memref.cast` operation.
pub fn cast<'c>(
    value: Value<'c, '_>,
    r#type: MemRefType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.cast", location)
        .add_operands(&[value])
        .add_results(&[r#type.into()])
        .build()
        .expect("valid operation")
}

/// Create a `memref.dealloc` operation.
pub fn dealloc<'c>(value: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.dealloc", location)
        .add_operands(&[value])
        .build()
        .expect("valid operation")
}

/// Create a `memref.dim` operation.
pub fn dim<'c>(
    value: Value<'c, '_>,
    index: Value<'c, '_>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.dim", location)
        .add_operands(&[value, index])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Create a `memref.get_global` operation.
pub fn get_global<'c>(
    context: &'c Context,
    name: &str,
    r#type: MemRefType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.get_global", location)
        .add_attributes(&[(
            Identifier::new(context, "name"),
            FlatSymbolRefAttribute::new(context, name).into(),
        )])
        .add_results(&[r#type.into()])
        .build()
        .expect("valid operation")
}

/// Create a `memref.view` operation.
pub fn view<'c>(
    context: &'c Context,
    source: Value<'c, '_>,
    byte_shift: Value<'c, '_>,
    sizes: &[Value<'c, '_>],
    result_type: MemRefType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.view", location)
        .add_operands(&[source])
        .add_operands(&[byte_shift])
        .add_operands(sizes)
        .add_results(&[result_type.into()])
        .add_attributes(&[(
            Identifier::new(context, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(context, &[1, 1, sizes.len() as i32]).into(),
        )])
        .build()
        .expect("valid operation")
}

/// Create a `memref.subview` operation.
#[allow(clippy::too_many_arguments)]
pub fn subview<'c>(
    context: &'c Context,
    source: Value<'c, '_>,
    offsets: &[Value<'c, '_>],
    sizes: &[Value<'c, '_>],
    strides: &[Value<'c, '_>],
    static_offsets: &[i64],
    static_sizes: &[i64],
    static_strides: &[i64],
    result_type: MemRefType<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.subview", location)
        .add_operands(&[source])
        .add_operands(offsets)
        .add_operands(sizes)
        .add_operands(strides)
        .add_results(&[result_type.into()])
        .add_attributes(&[
            (
                Identifier::new(context, "operand_segment_sizes"),
                DenseI32ArrayAttribute::new(
                    context,
                    &[
                        1,
                        offsets.len() as i32,
                        sizes.len() as i32,
                        strides.len() as i32,
                    ],
                )
                .into(),
            ),
            (
                Identifier::new(context, "static_offsets"),
                DenseI64ArrayAttribute::new(context, static_offsets).into(),
            ),
            (
                Identifier::new(context, "static_sizes"),
                DenseI64ArrayAttribute::new(context, static_sizes).into(),
            ),
            (
                Identifier::new(context, "static_strides"),
                DenseI64ArrayAttribute::new(context, static_strides).into(),
            ),
        ])
        .build()
        .expect("valid operation")
}

/// Create a `memref.global` operation.
#[allow(clippy::too_many_arguments)]
pub fn global<'c>(
    context: &'c Context,
    name: &str,
    visibility: Option<&str>,
    r#type: MemRefType<'c>,
    value: Option<Attribute<'c>>,
    constant: bool,
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.global", location).add_attributes(&[
        (
            Identifier::new(context, "sym_name"),
            StringAttribute::new(context, name).into(),
        ),
        (
            Identifier::new(context, "type"),
            TypeAttribute::new(r#type.into()).into(),
        ),
        (
            Identifier::new(context, "initial_value"),
            value.unwrap_or_else(|| Attribute::unit(context)),
        ),
    ]);

    if let Some(visibility) = visibility {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, visibility).into(),
        )]);
    }

    if constant {
        builder = builder.add_attributes(&[(
            Identifier::new(context, "constant"),
            Attribute::unit(context),
        )]);
    }

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.build().expect("valid operation")
}

/// Create a `memref.load` operation.
pub fn load<'c>(
    memref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.load", location)
        .add_operands(&[memref])
        .add_operands(indices)
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Create a `memref.rank` operation.
pub fn rank<'c>(value: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
    OperationBuilder::new("memref.rank", location)
        .add_operands(&[value])
        .enable_result_type_inference()
        .build()
        .expect("valid operation")
}

/// Create a `memref.store` operation.
pub fn store<'c>(
    value: Value<'c, '_>,
    memref: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.store", location)
        .add_operands(&[value, memref])
        .add_operands(indices)
        .build()
        .expect("valid operation")
}

/// Create a `memref.realloc` operation.
pub fn realloc<'c>(
    context: &'c Context,
    value: Value<'c, '_>,
    size: Option<Value<'c, '_>>,
    r#type: MemRefType<'c>,
    alignment: Option<IntegerAttribute<'c>>,
    location: Location<'c>,
) -> Operation<'c> {
    let mut builder = OperationBuilder::new("memref.realloc", location)
        .add_operands(&[value])
        .add_results(&[r#type.into()]);

    if let Some(size) = size {
        builder = builder.add_operands(&[size]);
    }

    if let Some(alignment) = alignment {
        builder =
            builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
    }

    builder.build().expect("valid operation")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::{func, index},
        ir::{
            attribute::{DenseElementsAttribute, StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::{FunctionType, IntegerType, RankedTensorType},
            Block, Module, Region, RegionLike, Type,
        },
        test::create_test_context,
    };

    fn compile_operation(name: &str, context: &Context, build_block: impl Fn(&Block)) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        let function = {
            let block = Block::new(&[]);

            build_block(&block);
            block.append_operation(func::r#return(&[], location));

            let region = Region::new();
            region.append_block(block);

            func::func(
                context,
                StringAttribute::new(context, "foo"),
                TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
                region,
                &[],
                Location::unknown(context),
            )
        };

        module.body().append_operation(function);
        assert!(module.as_operation().verify());
        insta::assert_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_alloc_and_dealloc() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("alloc", &context, |block| {
            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(dealloc(memref.result(0).unwrap().into(), location));
        })
    }

    #[test]
    fn compile_alloc_and_realloc() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("realloc", &context, |block| {
            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[8], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(realloc(
                &context,
                memref.result(0).unwrap().into(),
                None,
                MemRefType::new(Type::index(&context), &[42], None, None),
                None,
                location,
            ));
        })
    }

    #[test]
    fn compile_alloca() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("alloca", &context, |block| {
            block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
        })
    }

    #[test]
    fn compile_cast() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("cast", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::float64(&context), &[42], None, None),
                &[],
                &[],
                None,
                location,
            ));

            block.append_operation(cast(
                memref.result(0).unwrap().into(),
                Type::parse(&context, "memref<?xf64>")
                    .unwrap()
                    .try_into()
                    .unwrap(),
                location,
            ));
        })
    }

    #[test]
    fn compile_dim() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("dim", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 0),
                location,
            ));

            block.append_operation(dim(
                memref.result(0).unwrap().into(),
                index.result(0).unwrap().into(),
                location,
            ));
        })
    }

    #[test]
    fn compile_get_global() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let mem_ref_type = MemRefType::new(Type::index(&context), &[], None, None);

        module.body().append_operation(global(
            &context,
            "foo",
            None,
            mem_ref_type,
            None,
            false,
            None,
            location,
        ));

        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "bar"),
            TypeAttribute::new(FunctionType::new(&context, &[], &[]).into()),
            {
                let block = Block::new(&[]);

                block.append_operation(get_global(&context, "foo", mem_ref_type, location));
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
    fn compile_global() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);

        module.body().append_operation(global(
            &context,
            "foo",
            None,
            MemRefType::new(Type::index(&context), &[], None, None),
            None,
            false,
            None,
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_global_with_options() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let module = Module::new(location);
        let r#type = IntegerType::new(&context, 64).into();

        module.body().append_operation(global(
            &context,
            "foo",
            Some("private"),
            MemRefType::new(r#type, &[], None, None),
            Some(
                DenseElementsAttribute::new(
                    RankedTensorType::new(&[], r#type, None).into(),
                    &[IntegerAttribute::new(r#type, 42).into()],
                )
                .unwrap()
                .into(),
            ),
            true,
            Some(IntegerAttribute::new(
                IntegerType::new(&context, 64).into(),
                8,
            )),
            location,
        ));

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(module.as_operation());
    }

    #[test]
    fn compile_load() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("load", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(load(memref.result(0).unwrap().into(), &[], location));
        })
    }

    #[test]
    fn compile_load_with_index() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("load_with_index", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 0),
                location,
            ));

            block.append_operation(load(
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                location,
            ));
        })
    }

    #[test]
    fn compile_rank() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("rank", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));
            block.append_operation(rank(memref.result(0).unwrap().into(), location));
        })
    }

    #[test]
    fn compile_store() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("store", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let value = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 42),
                location,
            ));

            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[],
                location,
            ));
        })
    }

    #[test]
    fn compile_store_with_index() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("store_with_index", &context, |block| {
            let memref = block.append_operation(alloca(
                &context,
                MemRefType::new(Type::index(&context), &[1], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let value = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 42),
                location,
            ));

            let index = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 0),
                location,
            ));

            block.append_operation(store(
                value.result(0).unwrap().into(),
                memref.result(0).unwrap().into(),
                &[index.result(0).unwrap().into()],
                location,
            ));
        })
    }

    #[test]
    fn compile_view() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("view", &context, |block| {
            let byte_type = IntegerType::new(&context, 8).into();

            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(byte_type, &[8], None, None),
                &[],
                &[],
                None,
                location,
            ));

            let byte_shift = block.append_operation(index::constant(
                &context,
                IntegerAttribute::new(Type::index(&context), 0),
                location,
            ));

            block.append_operation(view(
                &context,
                memref.result(0).unwrap().into(),
                byte_shift.result(0).unwrap().into(),
                &[],
                MemRefType::new(byte_type, &[1], None, None),
                location,
            ));
        });
    }

    #[test]
    fn compile_subview() {
        let context = create_test_context();
        let location = Location::unknown(&context);

        compile_operation("subview", &context, |block| {
            let memref = block.append_operation(alloc(
                &context,
                MemRefType::new(Type::index(&context), &[8, 8], None, None),
                &[],
                &[],
                None,
                location,
            ));

            block.append_operation(subview(
                &context,
                memref.result(0).unwrap().into(),
                &[],
                &[],
                &[],
                &[0, 0],
                &[4, 4],
                &[1, 1],
                MemRefType::new(
                    Type::index(&context),
                    &[4, 4],
                    // TODO Construct a strided attribute using the C API without parsing.
                    Some(Attribute::parse(&context, "strided<[8, 1]>").unwrap()),
                    None,
                ),
                location,
            ));
        });
    }
}
