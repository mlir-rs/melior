// spell-checker: disable
//! Linalg passes.

melior_macro::passes!(
    "Linalg",
    [
        mlirCreateLinalgConvertElementwiseToLinalgPass,
        mlirCreateLinalgConvertLinalgToAffineLoopsPass,
        mlirCreateLinalgConvertLinalgToLoopsPass,
        mlirCreateLinalgConvertLinalgToParallelLoopsPass,
        mlirCreateLinalgLinalgBlockPackMatmul,
        mlirCreateLinalgLinalgDetensorizePass,
        mlirCreateLinalgLinalgElementwiseOpFusionPass,
        mlirCreateLinalgLinalgFoldIntoElementwisePass,
        mlirCreateLinalgLinalgFoldUnitExtentDimsPass,
        mlirCreateLinalgLinalgGeneralizeNamedOpsPass,
        mlirCreateLinalgLinalgInlineScalarOperandsPass,
        mlirCreateLinalgLinalgMorphOpsPass,
        mlirCreateLinalgLinalgSpecializeGenericOpsPass,
        mlirCreateLinalgSimplifyDepthwiseConvPass,
    ]
);
