//! Transform passes.

melior_macro::passes!(
    "Transforms",
    [
        mlirCreateTransformsCSE,
        mlirCreateTransformsCanonicalizer,
        mlirCreateTransformsCompositeFixedPointPass,
        mlirCreateTransformsControlFlowSink,
        mlirCreateTransformsGenerateRuntimeVerification,
        mlirCreateTransformsInliner,
        mlirCreateTransformsLocationSnapshot,
        mlirCreateTransformsLoopInvariantCodeMotion,
        mlirCreateTransformsLoopInvariantSubsetHoisting,
        mlirCreateTransformsMem2Reg,
        mlirCreateTransformsPrintIRPass,
        mlirCreateTransformsPrintOpStats,
        mlirCreateTransformsRemoveDeadValues,
        mlirCreateTransformsSCCP,
        mlirCreateTransformsSROA,
        mlirCreateTransformsStripDebugInfo,
        mlirCreateTransformsSymbolDCE,
        mlirCreateTransformsSymbolPrivatize,
        mlirCreateTransformsTopologicalSort,
        mlirCreateTransformsViewOpGraph,
    ]
);
