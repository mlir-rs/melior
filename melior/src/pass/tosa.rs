// spell-checker: disable
//! TOSA passes.

melior_macro::passes!(
    "Tosa",
    [
        mlirCreateTosaTosaArithConstantToTosaConstPass,
        mlirCreateTosaTosaAttachTarget,
        mlirCreateTosaTosaConvertIntegerTypeToSignless,
        mlirCreateTosaTosaInferShapesPass,
        mlirCreateTosaTosaLayerwiseConstantFoldPass,
        mlirCreateTosaTosaMakeBroadcastablePass,
        mlirCreateTosaTosaNarrowF64ToF32Pass,
        mlirCreateTosaTosaNarrowI64ToI32Pass,
        mlirCreateTosaTosaOptionalDecompositionsPass,
        mlirCreateTosaTosaReduceTransposes,
        mlirCreateTosaTosaValidation,
    ]
);
