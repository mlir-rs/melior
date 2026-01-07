use crate::Error;

use super::{Type, TypeLike};
use mlir_sys::{
    mlirShapedTypeGetDimSize, mlirShapedTypeGetElementType, mlirShapedTypeGetRank,
    mlirShapedTypeHasRank,
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
    fn has_rank() {
        let context = Context::new();
        let element_type = Type::index(&context);

        assert!(MemRefType::new(element_type, &[], None, None).has_rank());
        assert!(MemRefType::new(element_type, &[0], None, None).has_rank(),);
        assert!(MemRefType::new(element_type, &[0, 0], None, None).has_rank(),);
    }
}
