// spell-checker: disable
//! LLVM passes.

melior_macro::passes!(
    "LLVM",
    [
        mlirCreateLLVMDIScopeForLLVMFuncOpPass,
        mlirCreateLLVMLLVMAddComdats,
        mlirCreateLLVMLLVMLegalizeForExportPass,
        mlirCreateLLVMLLVMRequestCWrappersPass,
        mlirCreateLLVMLLVMUseDefaultVisibilityPass,
        mlirCreateLLVMNVVMOptimizeForTargetPass,
    ]
);
