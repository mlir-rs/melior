melior_macro::passes!(
    "Transforms",
    [
        mlirCreateTransformsCSEPass,
        mlirCreateFuncDuplicateFunctionEliminationPass,
    ]
);

fn main() {}
