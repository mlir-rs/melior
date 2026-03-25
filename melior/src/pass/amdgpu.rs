//! AMDGPU passes.

melior_macro::passes!(
    "AMDGPU",
    [
        // spell-checker: disable-next-line
        mlirCreateAMDGPUAmdgpuEmulateAtomicsPass,
        // spell-checker: disable-next-line
        mlirCreateAMDGPUAmdgpuFoldMemRefOpsPass,
        // spell-checker: disable-next-line
        mlirCreateAMDGPUAmdgpuMaskedloadToLoadPass,
        // spell-checker: disable-next-line
        mlirCreateAMDGPUAmdgpuResolveStridedMetadataPass,
    ]
);
