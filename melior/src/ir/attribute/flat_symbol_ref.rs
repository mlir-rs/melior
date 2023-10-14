use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef};
use mlir_sys::{mlirFlatSymbolRefAttrGet, mlirFlatSymbolRefAttrGetValue, MlirAttribute};

/// A flat symbol ref attribute.
#[derive(Clone, Copy)]
pub struct FlatSymbolRefAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FlatSymbolRefAttribute<'c> {
    /// Creates a flat symbol ref attribute.
    pub fn new(context: &'c Context, symbol: &str) -> Self {
        unsafe {
            Self::from_raw(mlirFlatSymbolRefAttrGet(
                context.to_raw(),
                StringRef::from_str(context, symbol).to_raw(),
            ))
        }
    }

    pub fn value(&self) -> &str {
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
    use crate::test::create_test_context;

    use super::*;

    #[test]
    fn new() {
        let context = create_test_context();
        assert_eq!(FlatSymbolRefAttribute::new(&context, "foo").value(), "foo");
    }
}
