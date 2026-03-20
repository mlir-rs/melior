//! LLVM passes.

melior_macro::passes!(
    "LLVM",
    [
        mlirCreateLLVMDIScopeForLLVMFuncOpPass,
        // spell-checker: disable-next-line
        mlirCreateLLVMLLVMAddComdats,
        mlirCreateLLVMLLVMLegalizeForExportPass,
        mlirCreateLLVMLLVMRequestCWrappersPass,
        mlirCreateLLVMLLVMUseDefaultVisibilityPass,
        mlirCreateLLVMNVVMOptimizeForTargetPass,
    ]
);
