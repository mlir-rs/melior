//! Arm SME passes.

melior_macro::passes!(
    "ArmSME",
    [
        mlirCreateArmSMEEnableArmStreaming,
        mlirCreateArmSMEOuterProductFusion,
        mlirCreateArmSMETestTileAllocation,
        mlirCreateArmSMEVectorLegalization,
    ]
);
