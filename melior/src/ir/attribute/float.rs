use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
};
use mlir_sys::{mlirFloatAttrDoubleGet, mlirFloatAttrGetValueDouble, MlirAttribute};

/// A float attribute.
#[derive(Clone, Copy, Hash)]
pub struct FloatAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FloatAttribute<'c> {
    /// Creates a float attribute.
    pub fn new(context: &'c Context, r#type: Type<'c>, number: f64) -> Self {
        unsafe {
            Self::from_raw(mlirFloatAttrDoubleGet(
                context.to_raw(),
                r#type.to_raw(),
                number,
            ))
        }
    }

    /// Returns a value.
    pub fn value(&self) -> f64 {
        unsafe { mlirFloatAttrGetValueDouble(self.to_raw()) }
    }
}

attribute_traits!(FloatAttribute, is_float, "float");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn value() {
        let context = create_test_context();

        assert_eq!(
            FloatAttribute::new(&context, Type::float64(&context), 42.0).value(),
            42.0
        );
    }
}
