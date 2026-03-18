use super::{Attribute, AttributeLike};
use crate::{
    Context, Error,
    ir::{Location, Type, TypeLike},
};
use mlir_sys::{
    MlirAttribute, mlirFloatAttrDoubleGet, mlirFloatAttrDoubleGetChecked,
    mlirFloatAttrGetValueDouble,
};

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

    /// Creates a float attribute with verification. Returns `None` if the type
    /// is not a valid float type.
    pub fn checked(location: Location<'c>, r#type: Type<'c>, number: f64) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirFloatAttrDoubleGetChecked(
                location.to_raw(),
                r#type.to_raw(),
                number,
            ))
        }
    }

    /// Returns a value.
    pub fn value(&self) -> f64 {
        unsafe { mlirFloatAttrGetValueDouble(self.to_raw()) }
    }

    unsafe fn from_option_raw(raw: mlir_sys::MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }
}

attribute_traits!(FloatAttribute, is_float, "float");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::Location, test::create_test_context};

    #[test]
    fn value() {
        let context = create_test_context();

        assert_eq!(
            FloatAttribute::new(&context, Type::float64(&context), 42.0).value(),
            42.0
        );
    }

    #[test]
    fn checked_valid() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let attr = FloatAttribute::checked(location, Type::float64(&context), 3.14);

        assert!(attr.is_some());
        let val = attr.unwrap().value();
        assert!((val - 3.14).abs() < 1e-10);
    }

    #[test]
    fn checked_invalid_type() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        // integer type is not a valid float type, so checked should return None
        let integer_type = Type::index(&context);
        let attr = FloatAttribute::checked(location, integer_type, 1.0);

        assert!(attr.is_none());
    }
}
