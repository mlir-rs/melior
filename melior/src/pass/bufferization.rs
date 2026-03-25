//! Bufferization passes.

melior_macro::passes!(
    "Bufferization",
    [
        mlirCreateBufferizationBufferDeallocationSimplificationPass,
        mlirCreateBufferizationBufferHoistingPass,
        mlirCreateBufferizationBufferLoopHoistingPass,
        mlirCreateBufferizationBufferResultsToOutParamsPass,
        mlirCreateBufferizationDropEquivalentBufferResultsPass,
        mlirCreateBufferizationEmptyTensorEliminationPass,
        mlirCreateBufferizationEmptyTensorToAllocTensorPass,
        mlirCreateBufferizationLowerDeallocationsPass,
        mlirCreateBufferizationOneShotBufferizePass,
        mlirCreateBufferizationOptimizeAllocationLivenessPass,
        mlirCreateBufferizationOwnershipBasedBufferDeallocationPass,
        mlirCreateBufferizationPromoteBuffersToStackPass,
    ]
);
