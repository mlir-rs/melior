use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirArrayAttrGet, mlirArrayAttrGetElement, mlirArrayAttrGetNumElements,
};

/// An array attribute.
#[derive(Clone, Copy, Hash)]
pub struct ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> ArrayAttribute<'c> {
    /// Creates an array attribute.
    pub fn new(context: &'c Context, values: &[Attribute<'c>]) -> Self {
        unsafe {
            Self::from_raw(mlirArrayAttrGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const _ as *const _,
            ))
        }
    }

    /// Returns a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirArrayAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an element.
    pub fn element(&self, index: usize) -> Result<Attribute<'c>, Error> {
        if index < self.len() {
            Ok(unsafe {
                Attribute::from_raw(mlirArrayAttrGetElement(
                    self.attribute.to_raw(),
                    index as isize,
                ))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(ArrayAttribute, is_dense_i64_array, "dense i64 array");

#[cfg(test)]
mod tests {
    use crate::{
        ir::{Type, attribute::IntegerAttribute, r#type::IntegerType},
        test::create_test_context,
    };

    use super::*;

    #[test]
    fn element() {
        let context = create_test_context();
        let r#type = IntegerType::new(&context, 64).into();
        let attributes = [
            IntegerAttribute::new(r#type, 1).into(),
            IntegerAttribute::new(r#type, 2).into(),
            IntegerAttribute::new(r#type, 3).into(),
        ];

        let attribute = ArrayAttribute::new(&context, &attributes);

        assert_eq!(attribute.element(0).unwrap(), attributes[0]);
        assert_eq!(attribute.element(1).unwrap(), attributes[1]);
        assert_eq!(attribute.element(2).unwrap(), attributes[2]);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = ArrayAttribute::new(
            &context,
            &[IntegerAttribute::new(Type::index(&context), 1).into()],
        );

        assert_eq!(attribute.len(), 1);
    }
}
