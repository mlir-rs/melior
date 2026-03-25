// spell-checker: disable
//! SPIR-V passes.

melior_macro::passes!(
    "SPIRV",
    [
        mlirCreateSPIRVSPIRVCanonicalizeGLPass,
        mlirCreateSPIRVSPIRVCompositeTypeLayoutPass,
        mlirCreateSPIRVSPIRVLowerABIAttributesPass,
        mlirCreateSPIRVSPIRVReplicatedConstantCompositePass,
        mlirCreateSPIRVSPIRVRewriteInsertsPass,
        mlirCreateSPIRVSPIRVUnifyAliasedResourcePass,
        mlirCreateSPIRVSPIRVUpdateVCEPass,
        mlirCreateSPIRVSPIRVWebGPUPreparePass,
    ]
);
