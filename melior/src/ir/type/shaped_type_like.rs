use crate::Error;

use super::{Type, TypeLike};
use mlir_sys::{
    mlirShapedTypeGetDimSize, mlirShapedTypeGetDynamicStrideOrOffset, mlirShapedTypeGetElementType,
    mlirShapedTypeGetRank, mlirShapedTypeHasRank, mlirShapedTypeHasStaticShape,
    mlirShapedTypeIsDynamicSize, mlirShapedTypeIsDynamicStrideOrOffset,
    mlirShapedTypeIsStaticStrideOrOffset,
};

/// A dimension size of a shaped type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimSize {
    /// A statically known dimension size.
    Static(u64),
    /// A dynamic (unknown at compile time) dimension.
    Dynamic,
}

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
    fn dim_size(&self, index: usize) -> Result<DimSize, Error> {
        if index < self.rank() {
            let raw = unsafe { mlirShapedTypeGetDimSize(self.to_raw(), index as isize) };

            if unsafe { mlirShapedTypeIsDynamicSize(raw) } {
                Ok(DimSize::Dynamic)
            } else {
                Ok(DimSize::Static(raw as u64))
            }
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dimension size",
                value: unsafe { Type::from_raw(self.to_raw()) }.to_string(),
                index,
            })
        }
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
            MemRefType::new(Type::index(&context), &[42], None, None).dim_size(0),
            Ok(DimSize::Static(42))
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 0], None, None).dim_size(0),
            Ok(DimSize::Static(42))
        );
        assert_eq!(
            MemRefType::new(Type::index(&context), &[0, 42], None, None).dim_size(1),
            Ok(DimSize::Static(42))
        );
    }

    #[test]
    fn dim_size_dynamic() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[i64::MIN], None, None).dim_size(0),
            Ok(DimSize::Dynamic)
        );
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
