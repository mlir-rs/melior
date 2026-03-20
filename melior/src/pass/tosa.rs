//! TOSA passes.

melior_macro::passes!(
    "Tosa",
    [
        // spell-checker: disable-next-line
        mlirCreateTosaTosaArithConstantToTosaConstPass,
        mlirCreateTosaTosaAttachTarget,
        // spell-checker: disable-next-line
        mlirCreateTosaTosaConvertIntegerTypeToSignless,
        mlirCreateTosaTosaInferShapesPass,
        mlirCreateTosaTosaLayerwiseConstantFoldPass,
        // spell-checker: disable-next-line
        mlirCreateTosaTosaMakeBroadcastablePass,
        mlirCreateTosaTosaNarrowF64ToF32Pass,
        mlirCreateTosaTosaNarrowI64ToI32Pass,
        mlirCreateTosaTosaOptionalDecompositionsPass,
        mlirCreateTosaTosaReduceTransposes,
        mlirCreateTosaTosaValidation,
    ]
);
