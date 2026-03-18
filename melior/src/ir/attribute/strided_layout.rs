use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirStridedLayoutAttrGet, mlirStridedLayoutAttrGetNumStrides,
    mlirStridedLayoutAttrGetOffset, mlirStridedLayoutAttrGetStride,
};

/// A strided layout attribute.
#[derive(Clone, Copy, Hash)]
pub struct StridedLayoutAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> StridedLayoutAttribute<'c> {
    /// Creates a strided layout attribute.
    pub fn new(context: &'c Context, offset: i64, strides: &[i64]) -> Self {
        unsafe {
            Self::from_raw(mlirStridedLayoutAttrGet(
                context.to_raw(),
                offset,
                strides.len() as isize,
                strides.as_ptr(),
            ))
        }
    }

    /// Returns the offset.
    pub fn offset(&self) -> i64 {
        unsafe { mlirStridedLayoutAttrGetOffset(self.to_raw()) }
    }

    /// Returns the number of strides.
    pub fn stride_count(&self) -> usize {
        (unsafe { mlirStridedLayoutAttrGetNumStrides(self.to_raw()) }) as usize
    }

    /// Returns a stride at the given index.
    pub fn stride(&self, index: usize) -> Result<i64, Error> {
        if index < self.stride_count() {
            Ok(unsafe { mlirStridedLayoutAttrGetStride(self.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "stride",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(StridedLayoutAttribute, is_strided_layout, "strided layout");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn new() {
        let context = create_test_context();
        StridedLayoutAttribute::new(&context, 0, &[1, 2, 3]);
    }

    #[test]
    fn offset() {
        let context = create_test_context();
        let attribute = StridedLayoutAttribute::new(&context, 42, &[]);

        assert_eq!(attribute.offset(), 42);
    }

    #[test]
    fn stride_count() {
        let context = create_test_context();
        let attribute = StridedLayoutAttribute::new(&context, 0, &[1, 2, 3]);

        assert_eq!(attribute.stride_count(), 3);
    }

    #[test]
    fn stride() {
        let context = create_test_context();
        let attribute = StridedLayoutAttribute::new(&context, 0, &[4, 5, 6]);

        assert_eq!(attribute.stride(0).unwrap(), 4);
        assert_eq!(attribute.stride(1).unwrap(), 5);
        assert_eq!(attribute.stride(2).unwrap(), 6);
        assert!(matches!(
            attribute.stride(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }
}
