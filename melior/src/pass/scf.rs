//! SCF passes.

melior_macro::passes!(
    "SCF",
    [
        mlirCreateSCFSCFForallToForLoop,
        mlirCreateSCFSCFForallToParallelLoop,
        mlirCreateSCFSCFForLoopCanonicalization,
        mlirCreateSCFSCFForLoopPeeling,
        mlirCreateSCFSCFForLoopRangeFolding,
        mlirCreateSCFSCFForLoopSpecialization,
        mlirCreateSCFSCFForToWhileLoop,
        // spell-checker: disable-next-line
        mlirCreateSCFSCFParallelForToNestedFors,
        mlirCreateSCFSCFParallelLoopFusion,
        mlirCreateSCFSCFParallelLoopSpecialization,
        mlirCreateSCFSCFParallelLoopTiling,
        mlirCreateSCFTestSCFParallelLoopCollapsing,
    ]
);
