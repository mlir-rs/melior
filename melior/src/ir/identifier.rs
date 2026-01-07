use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    MlirIdentifier, mlirIdentifierEqual, mlirIdentifierGet, mlirIdentifierGetContext,
    mlirIdentifierStr,
};
use std::marker::PhantomData;

/// An identifier.
#[derive(Clone, Copy, Debug)]
pub struct Identifier<'c> {
    raw: MlirIdentifier,
    _context: PhantomData<&'c Context>,
}

impl<'c> Identifier<'c> {
    /// Creates an identifier.
    pub fn new(context: &'c Context, name: &str) -> Self {
        unsafe {
            Self::from_raw(mlirIdentifierGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
            ))
        }
    }

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirIdentifierGetContext(self.raw)) }
    }

    /// Converts an identifier into a string reference.
    pub fn as_string_ref(&self) -> StringRef<'_> {
        unsafe { StringRef::from_raw(mlirIdentifierStr(self.raw)) }
    }

    /// Creates a location from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirIdentifier) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts a location into a raw object.
    pub const fn to_raw(self) -> MlirIdentifier {
        self.raw
    }
}

impl PartialEq for Identifier<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.raw, other.raw) }
    }
}

impl Eq for Identifier<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Identifier::new(&Context::new(), "foo");
    }

    #[test]
    fn context() {
        Identifier::new(&Context::new(), "foo").context();
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(
            Identifier::new(&context, "foo"),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            Identifier::new(&context, "foo"),
            Identifier::new(&context, "bar")
        );
    }
}
