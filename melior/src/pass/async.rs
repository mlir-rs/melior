//! Async passes.

melior_macro::passes!(
    "Async",
    [
        mlirCreateAsyncAsyncFuncToAsyncRuntimePass,
        mlirCreateAsyncAsyncParallelForPass,
        mlirCreateAsyncAsyncRuntimePolicyBasedRefCountingPass,
        mlirCreateAsyncAsyncRuntimeRefCountingPass,
        mlirCreateAsyncAsyncRuntimeRefCountingOptPass,
        mlirCreateAsyncAsyncToAsyncRuntimePass,
    ]
);
