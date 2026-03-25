//! Math passes.

melior_macro::passes!(
    "Math",
    [
        mlirCreateMathMathExpandOpsPass,
        mlirCreateMathMathExtendToSupportedTypes,
        // spell-checker: disable-next-line
        mlirCreateMathMathSincosFusionPass,
        mlirCreateMathMathUpliftToFMA,
    ]
);
