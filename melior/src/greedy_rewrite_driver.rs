use crate::{
    Error,
    ir::{Module, OperationRef},
    logical_result::LogicalResult,
    rewrite_pattern::FrozenRewritePatternSet,
};
use mlir_sys::{
    MlirGreedyRewriteDriverConfig, MlirGreedyRewriteStrictness,
    MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
    MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
    MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS,
    MlirGreedySimplifyRegionLevel,
    MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE,
    MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
    MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
    mlirApplyPatternsAndFoldGreedily, mlirGreedyRewriteDriverConfigCreate,
    mlirGreedyRewriteDriverConfigDestroy, mlirGreedyRewriteDriverConfigEnableConstantCSE,
    mlirGreedyRewriteDriverConfigEnableFolding, mlirGreedyRewriteDriverConfigGetMaxIterations,
    mlirGreedyRewriteDriverConfigGetMaxNumRewrites,
    mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel,
    mlirGreedyRewriteDriverConfigGetStrictness,
    mlirGreedyRewriteDriverConfigGetUseTopDownTraversal,
    mlirGreedyRewriteDriverConfigIsConstantCSEEnabled,
    mlirGreedyRewriteDriverConfigIsFoldingEnabled, mlirGreedyRewriteDriverConfigSetMaxIterations,
    mlirGreedyRewriteDriverConfigSetMaxNumRewrites,
    mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel,
    mlirGreedyRewriteDriverConfigSetStrictness,
    mlirGreedyRewriteDriverConfigSetUseTopDownTraversal, mlirWalkAndApplyPatterns,
};

/// Strictness level for the greedy rewrite driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GreedyRewriteStrictness {
    AnyOp = MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP,
    ExistingAndNewOps =
        MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS,
    ExistingOps = MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS,
}

impl GreedyRewriteStrictness {
    fn from_raw(raw: MlirGreedyRewriteStrictness) -> Self {
        match raw {
            x if x == MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_ANY_OP => {
                Self::AnyOp
            }
            x if x
                == MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_AND_NEW_OPS =>
            {
                Self::ExistingAndNewOps
            }
            x if x == MlirGreedyRewriteStrictness_MLIR_GREEDY_REWRITE_STRICTNESS_EXISTING_OPS => {
                Self::ExistingOps
            }
            _ => unreachable!(),
        }
    }
}

/// Region simplification level for the greedy rewrite driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GreedySimplifyRegionLevel {
    Disabled = MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED,
    Normal = MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL,
    Aggressive = MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE,
}

impl GreedySimplifyRegionLevel {
    fn from_raw(raw: MlirGreedySimplifyRegionLevel) -> Self {
        match raw {
            x if x == MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_DISABLED => {
                Self::Disabled
            }
            x if x == MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_NORMAL => {
                Self::Normal
            }
            x if x
                == MlirGreedySimplifyRegionLevel_MLIR_GREEDY_SIMPLIFY_REGION_LEVEL_AGGRESSIVE =>
            {
                Self::Aggressive
            }
            _ => unreachable!(),
        }
    }
}

/// Configuration for the greedy rewrite driver.
pub struct GreedyRewriteDriverConfig {
    raw: MlirGreedyRewriteDriverConfig,
}

impl GreedyRewriteDriverConfig {
    /// Creates a greedy rewrite driver configuration with default settings.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirGreedyRewriteDriverConfigCreate() },
        }
    }

    /// Sets the maximum number of iterations. Use -1 for no limit.
    pub fn set_max_iterations(&self, max: i64) {
        unsafe { mlirGreedyRewriteDriverConfigSetMaxIterations(self.raw, max) }
    }

    /// Returns the maximum number of iterations.
    pub fn max_iterations(&self) -> i64 {
        unsafe { mlirGreedyRewriteDriverConfigGetMaxIterations(self.raw) }
    }

    /// Sets the maximum number of rewrites per iteration. Use -1 for no limit.
    pub fn set_max_num_rewrites(&self, max: i64) {
        unsafe { mlirGreedyRewriteDriverConfigSetMaxNumRewrites(self.raw, max) }
    }

    /// Returns the maximum number of rewrites per iteration.
    pub fn max_num_rewrites(&self) -> i64 {
        unsafe { mlirGreedyRewriteDriverConfigGetMaxNumRewrites(self.raw) }
    }

    /// Sets whether to use top-down traversal.
    pub fn set_use_top_down_traversal(&self, enabled: bool) {
        unsafe { mlirGreedyRewriteDriverConfigSetUseTopDownTraversal(self.raw, enabled) }
    }

    /// Returns whether top-down traversal is enabled.
    pub fn use_top_down_traversal(&self) -> bool {
        unsafe { mlirGreedyRewriteDriverConfigGetUseTopDownTraversal(self.raw) }
    }

    /// Enables or disables folding during greedy rewriting.
    pub fn enable_folding(&self, enabled: bool) {
        unsafe { mlirGreedyRewriteDriverConfigEnableFolding(self.raw, enabled) }
    }

    /// Returns whether folding is enabled.
    pub fn is_folding_enabled(&self) -> bool {
        unsafe { mlirGreedyRewriteDriverConfigIsFoldingEnabled(self.raw) }
    }

    /// Sets the strictness level.
    pub fn set_strictness(&self, strictness: GreedyRewriteStrictness) {
        unsafe {
            mlirGreedyRewriteDriverConfigSetStrictness(
                self.raw,
                strictness as MlirGreedyRewriteStrictness,
            )
        }
    }

    /// Returns the strictness level.
    pub fn strictness(&self) -> GreedyRewriteStrictness {
        GreedyRewriteStrictness::from_raw(unsafe {
            mlirGreedyRewriteDriverConfigGetStrictness(self.raw)
        })
    }

    /// Sets the region simplification level.
    pub fn set_region_simplification_level(&self, level: GreedySimplifyRegionLevel) {
        unsafe {
            mlirGreedyRewriteDriverConfigSetRegionSimplificationLevel(
                self.raw,
                level as MlirGreedySimplifyRegionLevel,
            )
        }
    }

    /// Returns the region simplification level.
    pub fn region_simplification_level(&self) -> GreedySimplifyRegionLevel {
        GreedySimplifyRegionLevel::from_raw(unsafe {
            mlirGreedyRewriteDriverConfigGetRegionSimplificationLevel(self.raw)
        })
    }

    /// Enables or disables constant CSE.
    pub fn enable_constant_cse(&self, enabled: bool) {
        unsafe { mlirGreedyRewriteDriverConfigEnableConstantCSE(self.raw, enabled) }
    }

    /// Returns whether constant CSE is enabled.
    pub fn is_constant_cse_enabled(&self) -> bool {
        unsafe { mlirGreedyRewriteDriverConfigIsConstantCSEEnabled(self.raw) }
    }

    /// Converts the config into a raw object.
    pub const fn to_raw(&self) -> MlirGreedyRewriteDriverConfig {
        self.raw
    }
}

impl Default for GreedyRewriteDriverConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GreedyRewriteDriverConfig {
    fn drop(&mut self) {
        unsafe { mlirGreedyRewriteDriverConfigDestroy(self.raw) }
    }
}

/// Applies patterns and folds greedily to the given module.
///
/// The `patterns` argument is consumed (its ownership is transferred to the C
/// layer).
pub fn apply_patterns_and_fold_greedily(
    module: &Module,
    patterns: FrozenRewritePatternSet,
    config: &GreedyRewriteDriverConfig,
) -> Result<(), Error> {
    let result = LogicalResult::from_raw(unsafe {
        mlirApplyPatternsAndFoldGreedily(module.to_raw(), patterns.into_raw(), config.to_raw())
    });

    if result.is_success() {
        Ok(())
    } else {
        Err(Error::ApplyPatterns)
    }
}

/// Walks the operation and applies patterns using a fast walk-based driver.
///
/// The `patterns` argument is consumed.
pub fn walk_and_apply_patterns(op: OperationRef, patterns: FrozenRewritePatternSet) {
    unsafe { mlirWalkAndApplyPatterns(op.to_raw(), patterns.into_raw()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Location, Module},
        rewrite_pattern::RewritePatternSet,
        test::create_test_context,
    };

    #[test]
    fn new_config() {
        GreedyRewriteDriverConfig::new();
    }

    #[test]
    fn config_max_iterations() {
        let config = GreedyRewriteDriverConfig::new();

        config.set_max_iterations(10);

        assert_eq!(config.max_iterations(), 10);
    }

    #[test]
    fn config_strictness() {
        let config = GreedyRewriteDriverConfig::new();

        config.set_strictness(GreedyRewriteStrictness::ExistingOps);

        assert_eq!(config.strictness(), GreedyRewriteStrictness::ExistingOps);
    }

    #[test]
    fn config_max_num_rewrites() {
        let config = GreedyRewriteDriverConfig::new();

        config.set_max_num_rewrites(42);

        assert_eq!(config.max_num_rewrites(), 42);
    }

    #[test]
    fn config_use_top_down_traversal() {
        let config = GreedyRewriteDriverConfig::new();

        config.set_use_top_down_traversal(true);
        assert!(config.use_top_down_traversal());

        config.set_use_top_down_traversal(false);
        assert!(!config.use_top_down_traversal());
    }

    #[test]
    fn config_folding() {
        let config = GreedyRewriteDriverConfig::new();

        config.enable_folding(false);
        assert!(!config.is_folding_enabled());

        config.enable_folding(true);
        assert!(config.is_folding_enabled());
    }

    #[test]
    fn config_region_simplification() {
        let config = GreedyRewriteDriverConfig::new();

        config.set_region_simplification_level(GreedySimplifyRegionLevel::Disabled);
        assert_eq!(
            config.region_simplification_level(),
            GreedySimplifyRegionLevel::Disabled
        );

        config.set_region_simplification_level(GreedySimplifyRegionLevel::Normal);
        assert_eq!(
            config.region_simplification_level(),
            GreedySimplifyRegionLevel::Normal
        );

        config.set_region_simplification_level(GreedySimplifyRegionLevel::Aggressive);
        assert_eq!(
            config.region_simplification_level(),
            GreedySimplifyRegionLevel::Aggressive
        );
    }

    #[test]
    fn config_constant_cse() {
        let config = GreedyRewriteDriverConfig::new();

        config.enable_constant_cse(false);
        assert!(!config.is_constant_cse_enabled());

        config.enable_constant_cse(true);
        assert!(config.is_constant_cse_enabled());
    }

    #[test]
    fn apply_frozen_patterns() {
        let context = create_test_context();
        let module = Module::new(Location::unknown(&context));
        let patterns = RewritePatternSet::new(&context);
        let frozen = patterns.freeze();
        let config = GreedyRewriteDriverConfig::new();

        assert!(apply_patterns_and_fold_greedily(&module, frozen, &config).is_ok());
    }
}
