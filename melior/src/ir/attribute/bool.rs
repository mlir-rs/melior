use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{MlirAttribute, mlirBoolAttrGet, mlirBoolAttrGetValue};

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

attribute_traits!(BoolAttribute, is_bool, "bool");

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

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute = BoolAttribute::new(&context, true).into();

        assert!(BoolAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(BoolAttribute::try_from(attribute).is_err());
    }
}
