//! Arith passes.

melior_macro::passes!(
    "Arith",
    [
        mlirCreateArithArithEmulateUnsupportedFloats,
        mlirCreateArithArithEmulateWideInt,
        mlirCreateArithArithExpandOpsPass,
        mlirCreateArithArithIntRangeNarrowing,
        mlirCreateArithArithIntRangeOpts,
        mlirCreateArithArithUnsignedWhenEquivalentPass,
    ]
);
