use crate::Error;

use super::{Type, TypeLike};
use mlir_sys::{
    mlirShapedTypeGetDimSize, mlirShapedTypeGetDynamicSize, mlirShapedTypeGetDynamicStrideOrOffset,
    mlirShapedTypeGetElementType, mlirShapedTypeGetRank, mlirShapedTypeHasRank,
    mlirShapedTypeHasStaticShape, mlirShapedTypeIsDynamicDim, mlirShapedTypeIsDynamicSize,
    mlirShapedTypeIsDynamicStrideOrOffset, mlirShapedTypeIsStaticDim, mlirShapedTypeIsStaticSize,
    mlirShapedTypeIsStaticStrideOrOffset,
};

/// Trait for shaped types.
pub trait ShapedTypeLike<'c>: TypeLike<'c> {
    /// Returns a element type.
    fn element(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirShapedTypeGetElementType(self.to_raw())) }
    }

    /// Returns a rank.
    fn rank(&self) -> usize {
        (unsafe { mlirShapedTypeGetRank(self.to_raw()) }) as usize
    }

    /// Returns a dimension size.
    fn dim_size(&self, index: usize) -> Result<usize, Error> {
        if index < self.rank() {
            Ok((unsafe { mlirShapedTypeGetDimSize(self.to_raw(), index as isize) }) as usize)
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: unsafe { Type::from_raw(self.to_raw()) }.to_string(),
                index,
            })
        }
    }

    /// Returns a dimension size as a signed integer.
    ///
    /// Unlike [`dim_size`](Self::dim_size), this returns the raw `i64` value
    /// from the C API, where a negative value indicates a dynamic dimension.
    /// Use [`is_dynamic_dim`](Self::is_dynamic_dim) or
    /// [`is_dynamic_size`](Self::is_dynamic_size) to check for dynamic
    /// dimensions.
    fn dim_size_signed(&self, index: usize) -> Result<i64, Error> {
        if index < self.rank() {
            Ok(unsafe { mlirShapedTypeGetDimSize(self.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: unsafe { Type::from_raw(self.to_raw()) }.to_string(),
                index,
            })
        }
    }

    /// Checks if a dimension is dynamic.
    fn is_dynamic_dim(&self, index: usize) -> Result<bool, Error> {
        if index < self.rank() {
            Ok(unsafe { mlirShapedTypeIsDynamicDim(self.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: unsafe { Type::from_raw(self.to_raw()) }.to_string(),
                index,
            })
        }
    }

    /// Checks if a dimension is static.
    fn is_static_dim(&self, index: usize) -> Result<bool, Error> {
        if index < self.rank() {
            Ok(unsafe { mlirShapedTypeIsStaticDim(self.to_raw(), index as isize) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: unsafe { Type::from_raw(self.to_raw()) }.to_string(),
                index,
            })
        }
    }

    /// Checks if a size value represents a dynamic dimension.
    fn is_dynamic_size(size: i64) -> bool {
        unsafe { mlirShapedTypeIsDynamicSize(size) }
    }

    /// Checks if a size value represents a static dimension.
    fn is_static_size(size: i64) -> bool {
        unsafe { mlirShapedTypeIsStaticSize(size) }
    }

    /// Returns the sentinel value for a dynamic dimension size.
    fn dynamic_size() -> i64 {
        unsafe { mlirShapedTypeGetDynamicSize() }
    }

    /// Checks if a type has a static shape (all dimensions are static).
    fn has_static_shape(&self) -> bool {
        unsafe { mlirShapedTypeHasStaticShape(self.to_raw()) }
    }

    /// Checks if a value represents a dynamic stride or offset.
    fn is_dynamic_stride_or_offset(value: i64) -> bool {
        unsafe { mlirShapedTypeIsDynamicStrideOrOffset(value) }
    }

    /// Checks if a value represents a static stride or offset.
    fn is_static_stride_or_offset(value: i64) -> bool {
        unsafe { mlirShapedTypeIsStaticStrideOrOffset(value) }
    }

    /// Returns the sentinel value for a dynamic stride or offset.
    fn dynamic_stride_or_offset() -> i64 {
        unsafe { mlirShapedTypeGetDynamicStrideOrOffset() }
    }

    /// Checks if a type has a rank.
    fn has_rank(&self) -> bool {
        unsafe { mlirShapedTypeHasRank(self.to_raw()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        ir::{Type, r#type::MemRefType},
    };

    #[test]
    fn element() {
        let context = Context::new();
        let element_type = Type::index(&context);

        assert_eq!(
            MemRefType::new(element_type, &[], None, None).element(),
            element_type
        );
    }

    #[test]
    fn rank() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[], None, None).rank(),
            0
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0], None, None).rank(),
            1
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0, 0], None, None).rank(),
            2
        );
    }

    #[test]
    fn dim_size() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[], None, None).dim_size(0),
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: "memref<index>".into(),
                index: 0
            })
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42], None, None)
                .dim_size(0)
                .unwrap(),
            42
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 0], None, None)
                .dim_size(0)
                .unwrap(),
            42
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0, 42], None, None)
                .dim_size(1)
                .unwrap(),
            42
        );
    }

    #[test]
    fn dim_size_signed() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[42], None, None)
                .dim_size_signed(0)
                .unwrap(),
            42
        );

        // Dynamic dimension returns a negative value
        let dynamic = MemRefType::new(Type::index(&context), &[i64::MIN], None, None);
        assert!(dynamic.dim_size_signed(0).unwrap() < 0);
    }

    #[test]
    fn is_dynamic_dim() {
        let context = Context::new();

        assert!(
            !MemRefType::new(Type::index(&context), &[42], None, None)
                .is_dynamic_dim(0)
                .unwrap()
        );

        assert!(
            MemRefType::new(Type::index(&context), &[i64::MIN], None, None)
                .is_dynamic_dim(0)
                .unwrap()
        );
    }

    #[test]
    fn is_dynamic_size() {
        assert!(!MemRefType::is_dynamic_size(42));
        assert!(MemRefType::is_dynamic_size(
            MemRefType::new(Type::index(&Context::new()), &[i64::MIN], None, None,)
                .dim_size_signed(0)
                .unwrap()
        ));
    }

    #[test]
    fn is_static_dim() {
        let context = Context::new();

        assert!(
            MemRefType::new(Type::index(&context), &[42], None, None)
                .is_static_dim(0)
                .unwrap()
        );

        assert!(
            !MemRefType::new(Type::index(&context), &[i64::MIN], None, None)
                .is_static_dim(0)
                .unwrap()
        );
    }

    #[test]
    fn is_static_size() {
        assert!(MemRefType::is_static_size(42));
        assert!(!MemRefType::is_static_size(MemRefType::dynamic_size()));
    }

    #[test]
    fn dynamic_size() {
        let sentinel = MemRefType::dynamic_size();
        assert!(MemRefType::is_dynamic_size(sentinel));
        assert!(!MemRefType::is_static_size(sentinel));
    }

    #[test]
    fn has_static_shape() {
        let context = Context::new();

        assert!(MemRefType::new(Type::index(&context), &[42, 10], None, None).has_static_shape());

        assert!(
            !MemRefType::new(Type::index(&context), &[42, i64::MIN], None, None).has_static_shape()
        );
    }

    #[test]
    fn is_dynamic_stride_or_offset() {
        let sentinel = MemRefType::dynamic_stride_or_offset();
        assert!(MemRefType::is_dynamic_stride_or_offset(sentinel));
        assert!(!MemRefType::is_dynamic_stride_or_offset(0));
    }

    #[test]
    fn is_static_stride_or_offset() {
        assert!(MemRefType::is_static_stride_or_offset(0));
        assert!(!MemRefType::is_static_stride_or_offset(
            MemRefType::dynamic_stride_or_offset()
        ));
    }

    #[test]
    fn dynamic_stride_or_offset() {
        let sentinel = MemRefType::dynamic_stride_or_offset();
        assert!(MemRefType::is_dynamic_stride_or_offset(sentinel));
        assert!(!MemRefType::is_static_stride_or_offset(sentinel));
    }

    #[test]
    fn has_rank() {
        let context = Context::new();
        let element_type = Type::index(&context);

        assert!(MemRefType::new(element_type, &[], None, None).has_rank());
        assert!(MemRefType::new(element_type, &[0], None, None).has_rank(),);
        assert!(MemRefType::new(element_type, &[0, 0], None, None).has_rank(),);
    }
}
