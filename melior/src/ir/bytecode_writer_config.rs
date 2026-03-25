use mlir_sys::{
    MlirBytecodeWriterConfig, mlirBytecodeWriterConfigCreate,
    mlirBytecodeWriterConfigDesiredEmitVersion, mlirBytecodeWriterConfigDestroy,
};

/// A configuration for the bytecode writer.
pub struct BytecodeWriterConfig {
    raw: MlirBytecodeWriterConfig,
}

impl BytecodeWriterConfig {
    /// Creates a bytecode writer configuration.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirBytecodeWriterConfigCreate() },
        }
    }

    /// Sets the desired bytecode emit version.
    pub fn set_desired_emit_version(&self, version: i64) {
        unsafe { mlirBytecodeWriterConfigDesiredEmitVersion(self.raw, version) }
    }

    /// Converts the config into a raw object.
    pub const fn to_raw(&self) -> MlirBytecodeWriterConfig {
        self.raw
    }
}

impl Default for BytecodeWriterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BytecodeWriterConfig {
    fn drop(&mut self) {
        unsafe { mlirBytecodeWriterConfigDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        ir::{Location, Module, operation::OperationLike},
    };

    #[test]
    fn new() {
        BytecodeWriterConfig::new();
    }

    #[test]
    fn set_desired_emit_version() {
        let config = BytecodeWriterConfig::new();

        config.set_desired_emit_version(1);
    }

    #[test]
    fn write_bytecode() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));

        let bytes = module.as_operation().write_bytecode();

        assert!(!bytes.is_empty());
    }

    #[test]
    fn write_bytecode_with_config() {
        let context = Context::new();
        let module = Module::new(Location::unknown(&context));
        let config = BytecodeWriterConfig::new();

        let bytes = module.as_operation().write_bytecode_with_config(&config);

        assert!(bytes.is_ok());
        assert!(!bytes.unwrap().is_empty());
    }
}
