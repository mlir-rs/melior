// spell-checker: disable
//! TOSA passes.

melior_macro::passes!(
    "Tosa",
    [
        mlirCreateTosaTosaArithConstantToTosaConstPass,
        mlirCreateTosaTosaAttachTarget,
        mlirCreateTosaTosaConvertIntegerTypeToSignless,
        mlirCreateTosaTosaDowngrade1p1To1p0Pass,
        mlirCreateTosaTosaInferShapesPass,
        mlirCreateTosaTosaInputShape,
        mlirCreateTosaTosaLayerwiseConstantFoldPass,
        mlirCreateTosaTosaMakeBroadcastablePass,
        mlirCreateTosaTosaNarrowF64ToF32Pass,
        mlirCreateTosaTosaNarrowI64ToI32Pass,
        mlirCreateTosaTosaOptionalDecompositionsPass,
        mlirCreateTosaTosaReduceTransposes,
        mlirCreateTosaTosaValidation,
    ]
);
