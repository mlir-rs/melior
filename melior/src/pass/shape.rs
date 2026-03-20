//! Shape passes.

melior_macro::passes!(
    "Shape",
    [
        mlirCreateShapeOutlineShapeComputationPass,
        mlirCreateShapeRemoveShapeConstraintsPass,
        mlirCreateShapeShapeToShapeLoweringPass,
    ]
);
