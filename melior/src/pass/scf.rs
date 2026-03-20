// spell-checker: disable
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
        mlirCreateSCFSCFParallelForToNestedFors,
        mlirCreateSCFSCFParallelLoopFusion,
        mlirCreateSCFSCFParallelLoopSpecialization,
        mlirCreateSCFSCFParallelLoopTiling,
        mlirCreateSCFTestSCFParallelLoopCollapsing,
    ]
);
