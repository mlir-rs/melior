//! Affine passes.

melior_macro::passes!(
    "Affine",
    [
        mlirCreateAffineAffineDataCopyGeneration,
        mlirCreateAffineAffineExpandIndexOps,
        // spell-checker: disable-next-line
        mlirCreateAffineAffineExpandIndexOpsAsAffine,
        // spell-checker: disable-next-line
        mlirCreateAffineAffineFoldMemRefAliasOps,
        mlirCreateAffineAffineLoopFusion,
        mlirCreateAffineAffineLoopInvariantCodeMotion,
        mlirCreateAffineAffineLoopNormalize,
        mlirCreateAffineAffineLoopTiling,
        mlirCreateAffineAffineLoopUnroll,
        mlirCreateAffineAffineLoopUnrollAndJam,
        mlirCreateAffineAffineParallelize,
        mlirCreateAffineAffinePipelineDataTransfer,
        mlirCreateAffineAffineScalarReplacement,
        mlirCreateAffineAffineVectorize,
        mlirCreateAffineLoopCoalescing,
        // spell-checker: disable-next-line
        mlirCreateAffineRaiseMemrefDialect,
        mlirCreateAffineSimplifyAffineMinMaxPass,
        mlirCreateAffineSimplifyAffineStructures,
    ]
);
