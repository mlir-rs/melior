use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseI32ArrayGet, mlirDenseI32ArrayGetElement,
};

/// A dense i32 array attribute.
#[derive(Clone, Copy, Hash)]
pub struct DenseI32ArrayAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DenseI32ArrayAttribute<'c> {
    /// Creates a dense i32 array attribute.
    pub fn new(context: &'c Context, values: &[i32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseI32ArrayGet(
                context.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Returns a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirDenseArrayGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an element.
    pub fn element(&self, index: usize) -> Result<i32, Error> {
        if index < self.len() {
            Ok(unsafe { mlirDenseI32ArrayGetElement(self.attribute.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "array element",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(
    DenseI32ArrayAttribute,
    is_dense_i32_array,
    "dense i32 array"
);

#[cfg(test)]
mod tests {
    use crate::test::create_test_context;

    use super::*;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.len(), 3);
    }
}
