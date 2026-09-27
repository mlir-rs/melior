//! Experimental dialect operations and their builders generated automatically
//! from TableGen files.

#[doc(hidden)]
pub mod __private {
    pub struct Set;
    pub struct Unset;
}

melior_macro::dialect! {
    name: "affine",
    files: ["IR/AffineOps.td", "TransformOps/AffineTransformOps.td", "IR/AffineMemoryOpInterfaces.td"],
    include_directories: ["mlir/Dialect/Affine"],
}

melior_macro::dialect! {
    name: "amdgpu",
    files: ["IR/AMDGPU.td", "Transforms/Passes.td"],
    include_directories: ["mlir/Dialect/AMDGPU"],
}

melior_macro::dialect! {
    name: "arith",
    files: ["mlir/Dialect/Arith/IR/ArithOps.td"],
}

melior_macro::dialect! {
    name: "arm_neon",
    files: ["mlir/Dialect/ArmNeon/ArmNeon.td"],
}

melior_macro::dialect! {
    name: "arm_sve",
    files: ["mlir/Dialect/ArmSVE/IR/ArmSVE.td"],
}

melior_macro::dialect! {
    name: "arm_sme",
    files: ["ArmSME.td", "ArmSMEOps.td", "ArmSMEIntrinsicOps.td"],
    include_directories: ["mlir/Dialect/ArmSME/IR"],
}

melior_macro::dialect! {
    name: "async",
    files: ["AsyncDialect.td", "AsyncOps.td", "AsyncTypes.td"],
    include_directories: ["mlir/Dialect/Async/IR"],
}

melior_macro::dialect! {
    name: "builtin",
    files: ["mlir/IR/BuiltinOps.td"],
}

melior_macro::dialect! {
    name: "bufferization",
    files: [
        "IR/BufferizationOps.td",
        "IR/AllocationOpInterface.td",
        "IR/BufferizationEnums.td",
        "IR/BufferizableOpInterface.td",
        "TransformOps/BufferizationTransformOps.td",
        "Transforms/Passes.td",
    ],
    include_directories: ["mlir/Dialect/Bufferization"],
}

melior_macro::dialect! {
    name: "complex",
    files: ["ComplexBase.td", "ComplexOps.td"],
    include_directories: ["mlir/Dialect/Complex/IR"],
}

melior_macro::dialect! {
    name: "cf",
    files: ["mlir/Dialect/ControlFlow/IR/ControlFlowOps.td"],
}

melior_macro::dialect! {
    name: "dlti",
    files: ["DLTI.td", "DLTIAttrs.td", "DLTIBase.td"],
    include_directories: ["mlir/Dialect/DLTI"]
}

melior_macro::dialect! {
    name: "func",
    files: ["IR/FuncOps.td", "TransformOps/FuncTransformOps.td", "Transforms/Passes.td"],
    include_directories: ["mlir/Dialect/Func"],
}

melior_macro::dialect! {
    name: "index",
    files: ["mlir/Dialect/Index/IR/IndexOps.td"],
}

melior_macro::dialect! {
    name: "irdl",
    files: ["IRDLOps.td", "IRDL.td"],
    include_directories: ["mlir/Dialect/IRDL/IR"],
}

melior_macro::dialect! {
    name: "llvm",
    // spell-checker: disable-next-line
    files: [
        "LLVMOps.td",
        "LLVMIntrinsicOps.td",
        "LLVMDialect.td",
        "LLVMInterfaces.td",
        "LLVMTypes.td",
        "LLVMOpBase.td",
        "LLVMAttrDefs.td",
        "BasicPtxBuilderInterface.td",
    ],
    include_directories: ["mlir/Dialect/LLVMIR"],
}

melior_macro::dialect! {
    name: "memref",
    files: ["mlir/Dialect/MemRef/IR/MemRefOps.td"],
}

melior_macro::dialect! {
    name: "scf",
    files: ["mlir/Dialect/SCF/IR/SCFOps.td"],
}

melior_macro::dialect! {
    name: "pdl",
    files: ["mlir/Dialect/PDL/IR/PDLOps.td"],
}

melior_macro::dialect! {
    name: "pdl_interp",
    files: ["mlir/Dialect/PDLInterp/IR/PDLInterpOps.td"],
}

melior_macro::dialect! {
    name: "math",
    files: ["mlir/Dialect/Math/IR/MathOps.td"],
}

melior_macro::dialect! {
    name: "gpu",
    files: ["mlir/Dialect/GPU/IR/GPUOps.td"],
}

melior_macro::dialect! {
    name: "linalg",
    files: [
        "IR/LinalgOps.td",
        "IR/LinalgStructuredOps.td",
        "IR/LinalgRelayoutOps.td",
    ],
    include_directories: ["mlir/Dialect/Linalg"],
}

melior_macro::dialect! {
    name: "quant",
    files: ["IR/QuantOps.td", "Transforms/Passes.td"],
    include_directories: ["mlir/Dialect/Quant"],
}

melior_macro::dialect! {
    name: "shape",
    files: ["mlir/Dialect/Shape/IR/ShapeOps.td"],
}

melior_macro::dialect! {
    name: "sparse_tensor",
    files: ["mlir/Dialect/SparseTensor/IR/SparseTensorOps.td"],
}

melior_macro::dialect! {
    name: "tensor",
    files: ["mlir/Dialect/Tensor/IR/TensorOps.td"],
}

melior_macro::dialect! {
    name: "tosa",
    files: ["mlir/Dialect/Tosa/IR/TosaOps.td"],
}

melior_macro::dialect! {
    name: "transform",
    files: ["mlir/Dialect/Transform/IR/TransformOps.td"],
}

melior_macro::dialect! {
    name: "vector",
    files: ["mlir/Dialect/Vector/IR/VectorOps.td"],
}

melior_macro::dialect! {
    name: "x86",
    files: ["mlir/Dialect/X86/X86.td"],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context, dialect,
        ir::{
            Block, BlockLike, Location, Module, Region, RegionLike, Type,
            attribute::{
                ArrayAttribute, Attribute, IntegerAttribute, StringAttribute, TypeAttribute,
            },
            operation::OperationLike,
            r#type::{FunctionType, IntegerType, MemRefType, RankedTensorType},
        },
        pass::{self, PassManager},
        test::create_test_context,
    };

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

    fn test_operation<'c>(
        name: &str,
        context: &'c Context,
        argument_types: &[Type<'c>],
        callback: impl FnOnce(&Block<'c>),
    ) {
        let location = Location::unknown(context);
        let mut module = Module::new(location);

        module.body().append_operation(
            func::func(
                context,
                {
                    let block = Block::new(
                        &argument_types
                            .iter()
                            .copied()
                            .map(|r#type| (r#type, location))
                            .collect::<Vec<_>>(),
                    );

                    callback(&block);

                    let region = Region::new();
                    region.append_block(block);
                    region
                },
                StringAttribute::new(context, "foo"),
                TypeAttribute::new(FunctionType::new(context, argument_types, &[]).into()),
                location,
            )
            .into(),
        );

        convert_module(context, &mut module);

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(name, module.as_operation());
    }

    #[test]
    fn compile_float_arithmetics() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::float32(&context);

        test_operation("addf", &context, &[r#type, r#type], |block| {
            let add = block.append_operation(
                arith::addf(
                    &context,
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                    location,
                )
                .into(),
            );
            let sub = block.append_operation(
                arith::subf(
                    &context,
                    add.result(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
                    location,
                )
                .into(),
            );
            let mul = block.append_operation(
                arith::mulf(
                    &context,
                    add.result(0).unwrap().into(),
                    sub.result(0).unwrap().into(),
                    location,
                )
                .into(),
            );
            block.append_operation(
                arith::divf(
                    &context,
                    mul.result(0).unwrap().into(),
                    sub.result(0).unwrap().into(),
                    location,
                )
                .into(),
            );

            block.append_operation(func::r#return(&context, &[], location).into());
        });
    }

    #[test]
    fn compile_arith_addf_builder_with_reverse_order() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::float32(&context);

        test_operation("addf_builder", &context, &[r#type, r#type], |block| {
            block.append_operation(
                arith::AddFOperationBuilder::new(&context, location)
                    .lhs(block.argument(0).unwrap().into())
                    .rhs(block.argument(1).unwrap().into())
                    .build()
                    .into(),
            );

            block.append_operation(func::r#return(&context, &[], location).into());
        });
    }

    #[test]
    fn compile_llvm_alloca() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let integer_type = IntegerType::new(&context, 64).into();

        test_operation("alloc", &context, &[integer_type], |block| {
            let alloca_size = block.argument(0).unwrap().into();

            block.append_operation(
                llvm::AllocaOperation::builder(&context, location)
                    .array_size(alloca_size)
                    .elem_type(TypeAttribute::new(integer_type))
                    .res(dialect::llvm::r#type::pointer(&context, 0))
                    .build()
                    .into(),
            );

            block.append_operation(func::r#return(&context, &[], location).into());
        });
    }

    #[test]
    fn compile_llvm_alloca_builder() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let integer_type = IntegerType::new(&context, 64).into();
        let ptr_type = dialect::llvm::r#type::pointer(&context, 0);

        test_operation("alloc_builder", &context, &[integer_type], |block| {
            let alloca_size = block.argument(0).unwrap().into();

            block.append_operation(
                llvm::AllocaOperationBuilder::new(&context, location)
                    .alignment(IntegerAttribute::new(integer_type, 8))
                    .elem_type(TypeAttribute::new(integer_type))
                    .array_size(alloca_size)
                    .res(ptr_type)
                    .build()
                    .into(),
            );

            block.append_operation(func::r#return(&context, &[], location).into());
        });
    }

    // Structured `linalg` operations are not lowered by `convert_module`, so
    // they are verified as they are built instead of being compiled.
    fn test_linalg_operation<'c>(
        name: &str,
        context: &'c Context,
        argument_types: &[Type<'c>],
        callback: impl FnOnce(&Block<'c>),
    ) {
        let location = Location::unknown(context);
        let module = Module::new(location);

        module.body().append_operation(
            func::func(
                context,
                {
                    let block = Block::new(
                        &argument_types
                            .iter()
                            .copied()
                            .map(|r#type| (r#type, location))
                            .collect::<Vec<_>>(),
                    );

                    callback(&block);
                    block.append_operation(func::r#return(context, &[], location).into());

                    let region = Region::new();
                    region.append_block(block);
                    region
                },
                StringAttribute::new(context, "foo"),
                TypeAttribute::new(FunctionType::new(context, argument_types, &[]).into()),
                location,
            )
            .into(),
        );

        assert!(module.as_operation().verify());
        insta::assert_snapshot!(name, module.as_operation());
    }

    #[test]
    fn build_linalg_matmul() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let element_type = Type::float32(&context);
        let memref_type = MemRefType::new(element_type, &[4, 4], None, None).into();

        test_linalg_operation(
            "linalg.matmul",
            &context,
            &[memref_type, memref_type, memref_type],
            |block| {
                let region = Region::new();
                let body = Block::new(&[
                    (element_type, location),
                    (element_type, location),
                    (element_type, location),
                ]);

                let product = body.append_operation(
                    arith::mulf(
                        &context,
                        body.argument(0).unwrap().into(),
                        body.argument(1).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                let sum = body.append_operation(
                    arith::addf(
                        &context,
                        body.argument(2).unwrap().into(),
                        product.result(0).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                body.append_operation(
                    linalg::r#yield(&context, &[sum.result(0).unwrap().into()], location).into(),
                );
                region.append_block(body);

                block.append_operation(
                    linalg::matmul(
                        &context,
                        &[],
                        &[
                            block.argument(0).unwrap().into(),
                            block.argument(1).unwrap().into(),
                        ],
                        &[block.argument(2).unwrap().into()],
                        region,
                        location,
                    )
                    .into(),
                );
            },
        );
    }

    #[test]
    fn build_linalg_matmul_on_tensors() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let element_type = Type::float32(&context);
        let tensor_type = RankedTensorType::new(&[4, 4], element_type, None).into();

        test_linalg_operation(
            "linalg.matmul_on_tensors",
            &context,
            &[tensor_type, tensor_type, tensor_type],
            |block| {
                let region = Region::new();
                let body = Block::new(&[
                    (element_type, location),
                    (element_type, location),
                    (element_type, location),
                ]);

                let product = body.append_operation(
                    arith::mulf(
                        &context,
                        body.argument(0).unwrap().into(),
                        body.argument(1).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                let sum = body.append_operation(
                    arith::addf(
                        &context,
                        body.argument(2).unwrap().into(),
                        product.result(0).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                body.append_operation(
                    linalg::r#yield(&context, &[sum.result(0).unwrap().into()], location).into(),
                );
                region.append_block(body);

                // On tensors, the output is only an initial value, and the
                // operation returns one result per tensor output.
                let matmul = block.append_operation(
                    linalg::matmul(
                        &context,
                        &[tensor_type],
                        &[
                            block.argument(0).unwrap().into(),
                            block.argument(1).unwrap().into(),
                        ],
                        &[block.argument(2).unwrap().into()],
                        region,
                        location,
                    )
                    .into(),
                );

                assert_eq!(matmul.result_count(), 1);
            },
        );
    }

    #[test]
    fn build_linalg_generic() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let element_type = Type::float32(&context);
        let memref_type = MemRefType::new(element_type, &[4, 4], None, None).into();

        let identity_map = Attribute::parse(&context, "affine_map<(d0, d1) -> (d0, d1)>").unwrap();
        let parallel = Attribute::parse(&context, "#linalg.iterator_type<parallel>").unwrap();

        test_linalg_operation(
            "linalg.generic",
            &context,
            &[memref_type, memref_type, memref_type],
            |block| {
                let region = Region::new();
                let body = Block::new(&[
                    (element_type, location),
                    (element_type, location),
                    (element_type, location),
                ]);

                let sum = body.append_operation(
                    arith::addf(
                        &context,
                        body.argument(0).unwrap().into(),
                        body.argument(1).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                body.append_operation(
                    linalg::r#yield(&context, &[sum.result(0).unwrap().into()], location).into(),
                );
                region.append_block(body);

                block.append_operation(
                    linalg::generic(
                        &context,
                        &[],
                        &[
                            block.argument(0).unwrap().into(),
                            block.argument(1).unwrap().into(),
                        ],
                        &[block.argument(2).unwrap().into()],
                        region,
                        ArrayAttribute::new(&context, &[identity_map, identity_map, identity_map]),
                        ArrayAttribute::new(&context, &[parallel, parallel]),
                        location,
                    )
                    .into(),
                );
            },
        );
    }

    #[test]
    fn build_linalg_add() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let element_type = Type::float32(&context);
        let memref_type = MemRefType::new(element_type, &[4, 4], None, None).into();

        test_linalg_operation(
            "linalg.add",
            &context,
            &[memref_type, memref_type, memref_type],
            |block| {
                let region = Region::new();
                let body = Block::new(&[
                    (element_type, location),
                    (element_type, location),
                    (element_type, location),
                ]);

                let sum = body.append_operation(
                    arith::addf(
                        &context,
                        body.argument(0).unwrap().into(),
                        body.argument(1).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                body.append_operation(
                    linalg::r#yield(&context, &[sum.result(0).unwrap().into()], location).into(),
                );
                region.append_block(body);

                block.append_operation(
                    linalg::add(
                        &context,
                        &[],
                        &[
                            block.argument(0).unwrap().into(),
                            block.argument(1).unwrap().into(),
                        ],
                        &[block.argument(2).unwrap().into()],
                        region,
                        location,
                    )
                    .into(),
                );
            },
        );
    }

    #[test]
    fn build_linalg_conv_2d_nhwc_hwcf() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let element_type = Type::float32(&context);
        let input_type = MemRefType::new(element_type, &[1, 8, 8, 3], None, None).into();
        let filter_type = MemRefType::new(element_type, &[3, 3, 3, 4], None, None).into();
        // A stride of 2 takes the 8x8 input with a 3x3 filter to a 3x3 output.
        let output_type = MemRefType::new(element_type, &[1, 3, 3, 4], None, None).into();

        test_linalg_operation(
            "linalg.conv_2d_nhwc_hwcf",
            &context,
            &[input_type, filter_type, output_type],
            |block| {
                let region = Region::new();
                let body = Block::new(&[
                    (element_type, location),
                    (element_type, location),
                    (element_type, location),
                ]);

                let product = body.append_operation(
                    arith::mulf(
                        &context,
                        body.argument(0).unwrap().into(),
                        body.argument(1).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                let sum = body.append_operation(
                    arith::addf(
                        &context,
                        body.argument(2).unwrap().into(),
                        product.result(0).unwrap().into(),
                        location,
                    )
                    .into(),
                );
                body.append_operation(
                    linalg::r#yield(&context, &[sum.result(0).unwrap().into()], location).into(),
                );
                region.append_block(body);

                let mut convolution = linalg::conv_2_d_nhwc_hwcf(
                    &context,
                    &[],
                    &[
                        block.argument(0).unwrap().into(),
                        block.argument(1).unwrap().into(),
                    ],
                    &[block.argument(2).unwrap().into()],
                    region,
                    location,
                );
                // Optional attributes are not parameters of the free function.
                convolution
                    .set_strides(Attribute::parse(&context, "dense<2> : tensor<2xi64>").unwrap());

                block.append_operation(convolution.into());
            },
        );
    }

    #[test]
    fn build_memref_alloc_with_dynamic_size() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let memref_type = Type::parse(&context, "memref<?x4xf32>").unwrap();

        test_operation("alloc_dynamic_size", &context, &[index_type], |block| {
            block.append_operation(
                memref::alloc(
                    &context,
                    memref_type,
                    &[block.argument(0).unwrap().into()],
                    &[],
                    location,
                )
                .into(),
            );

            block.append_operation(func::r#return(&context, &[], location).into());
        });
    }

    #[test]
    fn read_operand_groups_of_attribute_sized_operation() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);
        let memref_type = Type::parse(&context, "memref<?x?xf32>").unwrap();

        let block = Block::new(&[(index_type, location), (index_type, location)]);
        let operation = memref::alloc(
            &context,
            memref_type,
            &[
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
            ],
            &[],
            location,
        );

        assert_eq!(operation.dynamic_sizes().unwrap().count(), 2);
        assert_eq!(operation.symbol_operands().unwrap().count(), 0);
    }
}
