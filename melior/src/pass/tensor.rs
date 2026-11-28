//! Tensor passes.

melior_macro::passes!(
    "Tensor",
    [
        mlirCreateTensorFoldTensorSubsetOpsPass,
        // spell-checker: disable-next-line
        mlirCreateTensorScalarizeSingleElementTensorReturnPass,
    ]
);
