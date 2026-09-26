//! Transform passes.

melior_macro::passes!(
    "Transforms",
    [
        // spell-checker: disable-next-line
        mlirCreateTransformsBubbleDownMemorySpaceCasts,
        mlirCreateTransformsCSEPass,
        mlirCreateTransformsCanonicalizerPass,
        mlirCreateTransformsCompositeFixedPointPass,
        mlirCreateTransformsControlFlowSinkPass,
        mlirCreateTransformsGenerateRuntimeVerificationPass,
        mlirCreateTransformsInlinerPass,
        mlirCreateTransformsLocationSnapshot,
        mlirCreateTransformsLoopInvariantCodeMotionPass,
        mlirCreateTransformsLoopInvariantSubsetHoistingPass,
        mlirCreateTransformsMem2Reg,
        mlirCreateTransformsPrintIRPass,
        mlirCreateTransformsPrintOpStatsPass,
        mlirCreateTransformsRemoveDeadValuesPass,
        mlirCreateTransformsSCCPPass,
        mlirCreateTransformsSROA,
        mlirCreateTransformsStripDebugInfoPass,
        mlirCreateTransformsSymbolDCEPass,
        mlirCreateTransformsSymbolPrivatizePass,
        mlirCreateTransformsTopologicalSortPass,
        mlirCreateTransformsViewOpGraphPass,
    ]
);
