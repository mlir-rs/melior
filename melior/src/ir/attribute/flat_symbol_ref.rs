use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef};
use core::str::Utf8Error;
use mlir_sys::{MlirAttribute, mlirFlatSymbolRefAttrGet, mlirFlatSymbolRefAttrGetValue};

/// A flat symbol ref attribute.
#[derive(Clone, Copy, Hash)]
pub struct FlatSymbolRefAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FlatSymbolRefAttribute<'c> {
    /// Creates a flat symbol ref attribute.
    pub fn new(context: &'c Context, symbol: &str) -> Self {
        unsafe {
            Self::from_raw(mlirFlatSymbolRefAttrGet(
                context.to_raw(),
                StringRef::new(symbol).to_raw(),
            ))
        }
    }

    /// Returns a value.
    pub fn value(&self) -> Result<&'c str, Utf8Error> {
        unsafe { StringRef::from_raw(mlirFlatSymbolRefAttrGetValue(self.to_raw())) }.as_str()
    }
}

attribute_traits!(
    FlatSymbolRefAttribute,
    is_flat_symbol_ref,
    "flat symbol ref"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn new() {
        let context = create_test_context();

        FlatSymbolRefAttribute::new(&context, "foo");
    }

    #[test]
    fn value() {
        let context = create_test_context();
        let value = FlatSymbolRefAttribute::new(&context, "foo").value();

        assert_eq!(value, Ok("foo"));
    }

    #[test]
    fn fail_to_get_non_utf8_value() {
        let context = create_test_context();
        let attribute =
            FlatSymbolRefAttribute::try_from(Attribute::parse(&context, r#"@"\FF""#).unwrap())
                .unwrap();

        assert!(attribute.value().is_err());
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute = FlatSymbolRefAttribute::new(&context, "foo").into();

        assert!(FlatSymbolRefAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(FlatSymbolRefAttribute::try_from(attribute).is_err());
    }
}
