//! Shard passes.

melior_macro::passes!(
    "Shard",
    [
        mlirCreateShardPartition,
        mlirCreateShardShardSimplify,
        mlirCreateShardShardingPropagation,
    ]
);
