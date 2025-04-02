//! GPU passes.

melior_macro::passes!(
    "GPU",
    [
        // spell-checker: disable-next-line
        mlirCreateGPUGpuAsyncRegionPass,
        mlirCreateGPUGpuDecomposeMemrefsPass,
        mlirCreateGPUGpuEliminateBarriers,
        mlirCreateGPUGpuKernelOutlining,
        mlirCreateGPUGpuLaunchSinkIndexComputations,
        mlirCreateGPUGpuMapParallelLoopsPass,
        mlirCreateGPUGpuModuleToBinaryPass,
        mlirCreateGPUGpuNVVMAttachTarget,
        mlirCreateGPUGpuROCDLAttachTarget,
        mlirCreateGPUGpuSPIRVAttachTarget,
    ]
);
