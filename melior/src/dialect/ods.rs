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
    name: "amx",
    files: ["mlir/Dialect/AMX/AMX.td"],
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
    files: ["mlir/Dialect/Linalg/IR/LinalgOps.td"],
}

melior_macro::dialect! {
    name: "quant",
    files: ["mlir/Dialect/Quant/QuantOps.td"],
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
    name: "x86vector",
    files: ["mlir/Dialect/X86Vector/X86Vector.td"],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect,
        ir::{
            attribute::{IntegerAttribute, StringAttribute, TypeAttribute}, operation::OperationLike, r#type::{FunctionType, IntegerType}, Block, BlockLike, Location, Module, Region, RegionLike, Type
        },
        pass::{self, PassManager},
        test::create_test_context,
        Context,
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
    fn compile_arith_addf() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::float32(&context);

        test_operation("addf", &context, &[r#type, r#type], |block| {
            block.append_operation(
                arith::addf(
                    &context,
                    block.argument(0).unwrap().into(),
                    block.argument(1).unwrap().into(),
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
}
