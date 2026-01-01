use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef};
use mlir_sys::{MlirAttribute, mlirStringAttrGet, mlirStringAttrGetValue};

/// A string attribute.
#[derive(Clone, Copy, Hash)]
pub struct StringAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> StringAttribute<'c> {
    /// Creates a string attribute.
    pub fn new(context: &'c Context, string: &str) -> Self {
        unsafe {
            Self::from_raw(mlirStringAttrGet(
                context.to_raw(),
                StringRef::new(string).to_raw(),
            ))
        }
    }

    /// Returns a value.
    pub fn value(&self) -> &'c str {
        unsafe { StringRef::from_raw(mlirStringAttrGetValue(self.to_raw())) }
            .as_str()
            .unwrap()
    }
}

attribute_traits!(StringAttribute, is_string, "string");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn value() {
        let context = create_test_context();
        let value = StringAttribute::new(&context, "foo").value();

        assert_eq!(value, "foo");
    }
}
