//! Transform dialect passes.

melior_macro::passes!(
    "Transform",
    [
        mlirCreateTransformCheckUsesPass,
        mlirCreateTransformInferEffectsPass,
        mlirCreateTransformInterpreterPass,
        mlirCreateTransformPreloadLibraryPass,
    ]
);
