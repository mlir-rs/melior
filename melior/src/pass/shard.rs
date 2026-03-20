//! Shard passes.

melior_macro::passes!(
    "Shard",
    [mlirCreateShardPartition, mlirCreateShardShardingPropagation,]
);
