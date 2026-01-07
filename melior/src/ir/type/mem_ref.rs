use super::{TypeLike, shaped_type_like::ShapedTypeLike};
use crate::{
    Error,
    ir::{Attribute, Location, Type, affine_map::AffineMap, attribute::AttributeLike},
};
use mlir_sys::{
    MlirType, mlirMemRefTypeGet, mlirMemRefTypeGetAffineMap, mlirMemRefTypeGetChecked,
    mlirMemRefTypeGetLayout, mlirMemRefTypeGetMemorySpace,
};

/// A mem-ref type.
#[derive(Clone, Copy, Debug, Hash)]
pub struct MemRefType<'c> {
    r#type: Type<'c>,
}

impl<'c> MemRefType<'c> {
    /// Creates a mem-ref type.
    pub fn new(
        r#type: Type<'c>,
        dimensions: &[i64],
        layout: Option<Attribute<'c>>,
        memory_space: Option<Attribute<'c>>,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirMemRefTypeGet(
                r#type.to_raw(),
                dimensions.len() as _,
                dimensions.as_ptr() as *const _,
                layout.unwrap_or_else(|| Attribute::null()).to_raw(),
                memory_space.unwrap_or_else(|| Attribute::null()).to_raw(),
            ))
        }
    }

    /// Creates a mem-ref type with diagnostics.
    pub fn checked(
        location: Location<'c>,
        r#type: Type<'c>,
        dimensions: &[u64],
        layout: Attribute<'c>,
        memory_space: Attribute<'c>,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirMemRefTypeGetChecked(
                location.to_raw(),
                r#type.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                layout.to_raw(),
                memory_space.to_raw(),
            ))
        }
    }

    /// Returns a layout.
    pub fn layout(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirMemRefTypeGetLayout(self.r#type.to_raw())) }
    }

    /// Returns an affine map.
    pub fn affine_map(&self) -> AffineMap<'c> {
        unsafe { AffineMap::from_raw(mlirMemRefTypeGetAffineMap(self.r#type.to_raw())) }
    }

    /// Returns a memory space.
    pub fn memory_space(&self) -> Option<Attribute<'c>> {
        unsafe { Attribute::from_option_raw(mlirMemRefTypeGetMemorySpace(self.r#type.to_raw())) }
    }

    unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }
}

impl<'c> ShapedTypeLike<'c> for MemRefType<'c> {}

type_traits!(MemRefType, is_mem_ref, "mem ref");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        let context = Context::new();

        assert_eq!(
            Type::from(MemRefType::new(Type::float64(&context), &[42], None, None,)),
            Type::parse(&context, "memref<42xf64>").unwrap()
        );
    }

    #[test]
    fn dynamic_dimension() {
        let context = Context::new();

        assert_eq!(
            Type::from(MemRefType::new(
                Type::float64(&context),
                &[i64::MIN],
                None,
                None,
            )),
            Type::parse(&context, "memref<?xf64>").unwrap()
        );
    }

    #[test]
    fn layout() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,).layout(),
            Attribute::parse(&context, "affine_map<(d0, d1) -> (d0, d1)>").unwrap(),
        );
    }

    #[test]
    fn affine_map() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None,)
                .affine_map()
                .to_string(),
            "(d0, d1) -> (d0, d1)"
        );
    }

    #[test]
    fn memory_space() {
        let context = Context::new();

        assert_eq!(
            MemRefType::new(Type::index(&context), &[42, 42], None, None).memory_space(),
            None,
        );
    }
}
