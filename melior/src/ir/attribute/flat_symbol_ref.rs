use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef};
use mlir_sys::{mlirFlatSymbolRefAttrGet, mlirFlatSymbolRefAttrGetValue, MlirAttribute};

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
    pub fn value(&self) -> &'c str {
        unsafe { StringRef::from_raw(mlirFlatSymbolRefAttrGetValue(self.to_raw())) }
            .as_str()
            .unwrap()
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

        assert_eq!(value, "foo");
    }
}
