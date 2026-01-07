use super::{TypeLike, shaped_type_like::ShapedTypeLike};
use crate::{
    Error,
    ir::{Attribute, Location, Type, attribute::AttributeLike},
};
use mlir_sys::{
    MlirType, mlirRankedTensorTypeGet, mlirRankedTensorTypeGetChecked,
    mlirRankedTensorTypeGetEncoding,
};

/// A ranked tensor type.
#[derive(Clone, Copy, Debug, Hash)]
pub struct RankedTensorType<'c> {
    r#type: Type<'c>,
}

impl<'c> RankedTensorType<'c> {
    /// Creates a ranked tensor type.
    pub fn new(dimensions: &[u64], r#type: Type<'c>, encoding: Option<Attribute<'c>>) -> Self {
        unsafe {
            Self::from_raw(mlirRankedTensorTypeGet(
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                r#type.to_raw(),
                encoding.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }

    /// Creates a ranked type with diagnostics.
    pub fn checked(
        dimensions: &[u64],
        r#type: Type<'c>,
        encoding: Attribute<'c>,
        location: Location<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirRankedTensorTypeGetChecked(
                location.to_raw(),
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                r#type.to_raw(),
                encoding.to_raw(),
            ))
        }
    }

    /// Returns an encoding.
    pub fn encoding(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirRankedTensorTypeGetEncoding(self.r#type.to_raw())) }
    }

    unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }
}

impl<'c> ShapedTypeLike<'c> for RankedTensorType<'c> {}

type_traits!(RankedTensorType, is_ranked_tensor, "tensor");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();

        assert_eq!(
            Type::from(RankedTensorType::new(&[42], Type::float64(&context), None)),
            Type::parse(&context, "tensor<42xf64>").unwrap()
        );
    }

    #[test]
    fn encoding() {
        let context = Context::new();

        assert_eq!(
            RankedTensorType::new(&[42, 42], Type::index(&context), None).encoding(),
            None,
        );
    }
}
