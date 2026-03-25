//! EmitC passes.

melior_macro::passes!(
    "EmitC",
    [
        mlirCreateEmitCFormExpressionsPass,
        mlirCreateEmitCWrapFuncInClassPass,
    ]
);
