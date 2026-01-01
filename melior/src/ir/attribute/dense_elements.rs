use super::{Attribute, AttributeLike};
use crate::{
    Error,
    ir::{Type, TypeLike},
};
use mlir_sys::{
    MlirAttribute, mlirDenseElementsAttrGet, mlirDenseElementsAttrGetInt32Value,
    mlirDenseElementsAttrGetInt64Value, mlirElementsAttrGetNumElements,
};

/// A dense elements attribute.
#[derive(Clone, Copy, Hash)]
pub struct DenseElementsAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DenseElementsAttribute<'c> {
    /// Creates a dense elements attribute.
    pub fn new(r#type: Type<'c>, values: &[Attribute<'c>]) -> Result<Self, Error> {
        if r#type.is_shaped() {
            Ok(unsafe {
                Self::from_raw(mlirDenseElementsAttrGet(
                    r#type.to_raw(),
                    values.len() as isize,
                    values.as_ptr() as *const _ as *const _,
                ))
            })
        } else {
            Err(Error::TypeExpected("shaped", r#type.to_string()))
        }
    }

    /// Returns a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an i32 element.
    // TODO Prevent calling these type specific methods on other types.
    pub fn i32_element(&self, index: usize) -> Result<i32, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetInt32Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an i64 element.
    pub fn i64_element(&self, index: usize) -> Result<i64, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetInt64Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(DenseElementsAttribute, is_dense_elements, "dense elements");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            attribute::IntegerAttribute,
            r#type::{IntegerType, MemRefType},
        },
        test::create_test_context,
    };

    #[test]
    fn i32_element() {
        let context = create_test_context();
        let integer_type = IntegerType::new(&context, 32).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(integer_type, 42).into()],
        )
        .unwrap();

        assert_eq!(attribute.i32_element(0), Ok(42));
        assert_eq!(attribute.i32_element(1), Ok(42));
        assert_eq!(attribute.i32_element(2), Ok(42));
        assert_eq!(
            attribute.i32_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }

    #[test]
    fn i64_element() {
        let context = create_test_context();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(integer_type, 42).into()],
        )
        .unwrap();

        assert_eq!(attribute.i64_element(0), Ok(42));
        assert_eq!(attribute.i64_element(1), Ok(42));
        assert_eq!(attribute.i64_element(2), Ok(42));
        assert_eq!(
            attribute.i64_element(3),
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: attribute.to_string(),
                index: 3,
            })
        );
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let integer_type = IntegerType::new(&context, 64).into();
        let attribute = DenseElementsAttribute::new(
            MemRefType::new(integer_type, &[3], None, None).into(),
            &[IntegerAttribute::new(integer_type, 0).into()],
        )
        .unwrap();

        assert_eq!(attribute.len(), 3);
    }
}
