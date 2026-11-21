use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use core::marker::PhantomData;
use mlir_sys::{
    MlirIdentifier, mlirIdentifierEqual, mlirIdentifierGet, mlirIdentifierGetContext,
    mlirIdentifierStr,
};

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
    pub fn as_string_ref(&self) -> StringRef<'c> {
        unsafe { StringRef::from_raw(mlirIdentifierStr(self.raw)) }
    }

    /// Creates an identifier from a raw object.
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

    /// Converts an identifier into a raw object.
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
    fn as_string_ref() {
        let context = Context::new();

        assert_eq!(
            Identifier::new(&context, "foo").as_string_ref().as_str(),
            Ok("foo")
        );
    }

    #[test]
    fn as_string_ref_outliving_identifier() {
        let context = Context::new();
        let string = {
            let identifier = Identifier::new(&context, "foo");

            identifier.as_string_ref()
        };

        assert_eq!(string.as_str(), Ok("foo"));
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
