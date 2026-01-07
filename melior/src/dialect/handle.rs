use super::DialectRegistry;
use crate::{context::Context, dialect::Dialect, string_ref::StringRef};
use mlir_sys::{
    MlirDialectHandle, mlirDialectHandleGetNamespace, mlirDialectHandleInsertDialect,
    mlirDialectHandleLoadDialect, mlirDialectHandleRegisterDialect, mlirGetDialectHandle__async__,
    mlirGetDialectHandle__cf__, mlirGetDialectHandle__func__, mlirGetDialectHandle__gpu__,
    mlirGetDialectHandle__linalg__, mlirGetDialectHandle__llvm__, mlirGetDialectHandle__pdl__,
    mlirGetDialectHandle__quant__, mlirGetDialectHandle__scf__, mlirGetDialectHandle__shape__,
    mlirGetDialectHandle__sparse_tensor__, mlirGetDialectHandle__tensor__,
};

/// A dialect handle.
#[derive(Clone, Copy, Debug)]
pub struct DialectHandle {
    raw: MlirDialectHandle,
}

impl DialectHandle {
    /// Creates a `async` dialect handle.
    pub fn r#async() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__async__()) }
    }

    /// Creates a `cf` dialect handle.
    pub fn cf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__cf__()) }
    }

    /// Creates a `func` dialect handle.
    pub fn func() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__func__()) }
    }

    /// Creates a `gpu` dialect handle.
    pub fn gpu() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__gpu__()) }
    }

    /// Creates a `linalg` dialect handle.
    pub fn linalg() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__linalg__()) }
    }

    /// Creates a `llvm` dialect handle.
    pub fn llvm() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__llvm__()) }
    }

    /// Creates a `pdl` dialect handle.
    pub fn pdl() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__pdl__()) }
    }

    /// Creates a `quant` dialect handle.
    pub fn quant() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__quant__()) }
    }

    /// Creates a `scf` dialect handle.
    pub fn scf() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__scf__()) }
    }

    /// Creates a `shape` dialect handle.
    pub fn shape() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__shape__()) }
    }

    /// Creates a `sparse_tensor` dialect handle.
    pub fn sparse_tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__sparse_tensor__()) }
    }

    /// Creates a `tensor` dialect handle.
    pub fn tensor() -> Self {
        unsafe { Self::from_raw(mlirGetDialectHandle__tensor__()) }
    }

    /// Returns a namespace.
    pub fn namespace(&self) -> StringRef<'_> {
        unsafe { StringRef::from_raw(mlirDialectHandleGetNamespace(self.raw)) }
    }

    /// Inserts a dialect into a dialect registry.
    pub fn insert_dialect(&self, registry: &DialectRegistry) {
        unsafe { mlirDialectHandleInsertDialect(self.raw, registry.to_raw()) }
    }

    /// Loads a dialect into a context.
    pub fn load_dialect<'c>(&self, context: &'c Context) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirDialectHandleLoadDialect(self.raw, context.to_raw())) }
    }

    /// Registers a dialect into a context.
    pub fn register_dialect(&self, context: &Context) {
        unsafe { mlirDialectHandleRegisterDialect(self.raw, context.to_raw()) }
    }

    /// Creates a dialect handle from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(handle: MlirDialectHandle) -> Self {
        Self { raw: handle }
    }

    /// Converts a dialect handle into a raw object.
    pub const fn to_raw(self) -> MlirDialectHandle {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn func() {
        DialectHandle::func();
    }

    #[test]
    fn llvm() {
        DialectHandle::llvm();
    }

    #[test]
    fn namespace() {
        DialectHandle::func().namespace();
    }

    #[test]
    fn insert_dialect() {
        let registry = DialectRegistry::new();

        DialectHandle::func().insert_dialect(&registry);
    }

    #[test]
    fn load_dialect() {
        let context = Context::new();

        DialectHandle::func().load_dialect(&context);
    }

    #[test]
    fn register_dialect() {
        let context = Context::new();

        DialectHandle::func().register_dialect(&context);
    }
}
