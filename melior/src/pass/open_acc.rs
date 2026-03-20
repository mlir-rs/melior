//! OpenACC passes.

melior_macro::passes!(
    "OpenACC",
    [
        mlirCreateOpenACCACCIfClauseLowering,
        mlirCreateOpenACCACCImplicitData,
        mlirCreateOpenACCACCImplicitDeclare,
        mlirCreateOpenACCACCImplicitRoutine,
        mlirCreateOpenACCACCLegalizeSerial,
        mlirCreateOpenACCACCLoopTiling,
        mlirCreateOpenACCACCSpecializeForDevice,
        mlirCreateOpenACCACCSpecializeForHost,
        mlirCreateOpenACCLegalizeDataValuesInRegion,
        mlirCreateOpenACCOffloadLiveInValueCanonicalization,
    ]
);
