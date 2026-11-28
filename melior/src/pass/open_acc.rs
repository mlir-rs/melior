// spell-checker: disable
//! OpenACC passes.

melior_macro::passes!(
    "OpenACC",
    [
        mlirCreateOpenACCACCBindRoutine,
        mlirCreateOpenACCACCComputeLowering,
        mlirCreateOpenACCACCDeclareGPUModuleInsertion,
        mlirCreateOpenACCACCEmitRemarksLoop,
        mlirCreateOpenACCACCEmitRemarksPrivate,
        mlirCreateOpenACCACCIfClauseLowering,
        mlirCreateOpenACCACCImplicitData,
        mlirCreateOpenACCACCImplicitDeclare,
        mlirCreateOpenACCACCImplicitRoutine,
        mlirCreateOpenACCACCLegalizeSerial,
        mlirCreateOpenACCACCLoopTiling,
        mlirCreateOpenACCACCRecipeMaterialization,
        mlirCreateOpenACCACCRoutineLowering,
        mlirCreateOpenACCACCRoutineToGPUFunc,
        mlirCreateOpenACCACCSpecializeForDevice,
        mlirCreateOpenACCACCSpecializeForHost,
        mlirCreateOpenACCLegalizeDataValuesInRegion,
        mlirCreateOpenACCOffloadLiveInValueCanonicalization,
        mlirCreateOpenACCOffloadTargetVerifier,
    ]
);
