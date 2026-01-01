use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Error,
};
use mlir_sys::{mlirTypeAttrGet, mlirTypeAttrGetValue, MlirAttribute};

/// A type attribute.
#[derive(Clone, Copy, Hash)]
pub struct TypeAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> TypeAttribute<'c> {
    /// Creates a type attribute.
    pub fn new(r#type: Type<'c>) -> Self {
        unsafe { Self::from_raw(mlirTypeAttrGet(r#type.to_raw())) }
    }

    /// Returns a type value.
    pub fn value(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirTypeAttrGetValue(self.to_raw())) }
    }
}

attribute_traits!(TypeAttribute, is_type, "type");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn value() {
        let context = create_test_context();
        let r#type = Type::index(&context);

        assert_eq!(TypeAttribute::new(r#type).value(), r#type);
    }
}
