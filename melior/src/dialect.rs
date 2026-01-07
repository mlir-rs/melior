//! Dialect handles, instances, and registry.

pub mod arith;
pub mod cf;
pub mod func;
mod handle;
pub mod index;
pub mod llvm;
pub mod memref;
mod registry;
pub mod scf;

pub use self::{handle::DialectHandle, registry::DialectRegistry};
use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{MlirDialect, mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace};
use std::{marker::PhantomData, str::Utf8Error};

#[cfg(feature = "ods-dialects")]
pub mod ods;

/// A dialect.
#[derive(Clone, Copy, Debug)]
pub struct Dialect<'c> {
    raw: MlirDialect,
    _context: PhantomData<&'c Context>,
}

impl<'c> Dialect<'c> {
    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirDialectGetContext(self.raw)) }
    }

    /// Returns a namespace.
    pub fn namespace(&self) -> Result<&str, Utf8Error> {
        unsafe { StringRef::from_raw(mlirDialectGetNamespace(self.raw)) }.as_str()
    }

    /// Creates a dialect from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(dialect: MlirDialect) -> Self {
        Self {
            raw: dialect,
            _context: Default::default(),
        }
    }
}

impl PartialEq for Dialect<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.raw, other.raw) }
    }
}

impl Eq for Dialect<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace() {
        let context = Context::new();

        assert_eq!(
            DialectHandle::llvm()
                .load_dialect(&context)
                .namespace()
                .unwrap(),
            "llvm"
        );
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::func().load_dialect(&context)
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            DialectHandle::func().load_dialect(&context),
            DialectHandle::llvm().load_dialect(&context)
        );
    }
}
