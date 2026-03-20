//! MemRef passes.

melior_macro::passes!(
    "MemRef",
    [
        mlirCreateMemRefExpandOpsPass,
        // spell-checker: disable-next-line
        mlirCreateMemRefExpandReallocPass,
        mlirCreateMemRefExpandStridedMetadataPass,
        // spell-checker: disable-next-line
        mlirCreateMemRefFlattenMemrefsPass,
        // spell-checker: disable-next-line
        mlirCreateMemRefFoldMemRefAliasOpsPass,
        mlirCreateMemRefMemRefEmulateWideInt,
        // spell-checker: disable-next-line
        mlirCreateMemRefNormalizeMemRefsPass,
        mlirCreateMemRefReifyResultShapesPass,
        mlirCreateMemRefResolveRankedShapeTypeResultDimsPass,
        mlirCreateMemRefResolveShapedTypeResultDimsPass,
    ]
);
