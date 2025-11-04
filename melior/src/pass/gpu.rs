//! GPU passes.

melior_macro::passes!(
    "GPU",
    [
        // spell-checker: disable-next-line
        mlirCreateGPUGpuAsyncRegionPass,
        mlirCreateGPUGpuDecomposeMemrefsPass,
        mlirCreateGPUGpuEliminateBarriers,
        mlirCreateGPUGpuKernelOutliningPass,
        mlirCreateGPUGpuLaunchSinkIndexComputationsPass,
        mlirCreateGPUGpuMapParallelLoopsPass,
        mlirCreateGPUGpuModuleToBinaryPass,
        mlirCreateGPUGpuNVVMAttachTarget,
        mlirCreateGPUGpuROCDLAttachTarget,
        mlirCreateGPUGpuSPIRVAttachTarget,
    ]
);
