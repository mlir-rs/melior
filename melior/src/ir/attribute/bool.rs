use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{mlirBoolAttrGet, mlirBoolAttrGetValue, MlirAttribute};

/// A bool attribute.
#[derive(Clone, Copy, Hash)]
pub struct BoolAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> BoolAttribute<'c> {
    /// Creates a bool attribute.
    pub fn new(context: &'c Context, boolean: bool) -> Self {
        unsafe {
            Self::from_raw(mlirBoolAttrGet(
                context.to_raw(),
                if boolean { 1 } else { 0 },
            ))
        }
    }

    /// Returns a value.
    pub fn value(&self) -> bool {
        unsafe { mlirBoolAttrGetValue(self.to_raw()) }
    }
}

attribute_traits!(BoolAttribute, is_string, "string");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn value() {
        let context = create_test_context();
        let value = BoolAttribute::new(&context, true).value();

        assert!(value);
    }
}
