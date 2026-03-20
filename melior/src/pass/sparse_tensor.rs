//! Sparse tensor passes.

melior_macro::passes!(
    "SparseTensor",
    [
        mlirCreateSparseTensorLowerForeachToSCF,
        mlirCreateSparseTensorLowerSparseIterationToSCF,
        mlirCreateSparseTensorLowerSparseOpsToForeach,
        mlirCreateSparseTensorPreSparsificationRewrite,
        mlirCreateSparseTensorSparseAssembler,
        mlirCreateSparseTensorSparseBufferRewrite,
        mlirCreateSparseTensorSparseGPUCodegen,
        mlirCreateSparseTensorSparseReinterpretMap,
        mlirCreateSparseTensorSparseSpaceCollapse,
        mlirCreateSparseTensorSparseTensorCodegen,
        mlirCreateSparseTensorSparseTensorConversionPass,
        mlirCreateSparseTensorSparseVectorization,
        mlirCreateSparseTensorSparsificationAndBufferization,
        mlirCreateSparseTensorSparsificationPass,
        mlirCreateSparseTensorStageSparseOperations,
        mlirCreateSparseTensorStorageSpecifierToLLVM,
    ]
);
