//! Vector passes.

melior_macro::passes!(
    "Vector",
    [
        mlirCreateVectorLowerVectorMaskPass,
        mlirCreateVectorLowerVectorMultiReduction,
        mlirCreateVectorLowerVectorToFromElementsToShuffleTree,
    ]
);
