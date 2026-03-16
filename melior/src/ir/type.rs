//! Types and type IDs.

#[macro_use]
mod r#macro;
mod function;
pub mod id;
mod integer;
mod mem_ref;
mod ranked_tensor;
mod shaped_type_like;
mod tuple;
mod type_like;

pub use self::{
    function::FunctionType, id::TypeId, integer::IntegerType, mem_ref::MemRefType,
    ranked_tensor::RankedTensorType, shaped_type_like::ShapedTypeLike, tuple::TupleType,
    type_like::TypeLike,
};
use super::Location;
use crate::{context::Context, string_ref::StringRef, utility::print_callback};
use mlir_sys::{
    MlirType, mlirBF16TypeGet, mlirF16TypeGet, mlirF32TypeGet, mlirF64TypeGet,
    mlirFloat4E2M1FNTypeGet, mlirFloat6E2M3FNTypeGet, mlirFloat6E3M2FNTypeGet,
    mlirFloat8E3M4TypeGet, mlirFloat8E4M3B11FNUZTypeGet, mlirFloat8E4M3FNTypeGet,
    mlirFloat8E4M3FNUZTypeGet, mlirFloat8E4M3TypeGet, mlirFloat8E5M2FNUZTypeGet,
    mlirFloat8E5M2TypeGet, mlirFloat8E8M0FNUTypeGet, mlirFloatTypeGetWidth, mlirIndexTypeGet,
    mlirNoneTypeGet, mlirTypeEqual, mlirTypeParseGet, mlirTypePrint, mlirVectorTypeGet,
    mlirVectorTypeGetChecked, mlirVectorTypeGetScalable, mlirVectorTypeGetScalableChecked,
    mlirVectorTypeIsDimScalable, mlirVectorTypeIsScalable,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// A type.
// Types are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct Type<'c> {
    raw: MlirType,
    _context: PhantomData<&'c Context>,
}

impl<'c> Type<'c> {
    /// Parses a type.
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirTypeParseGet(
                context.to_raw(),
                StringRef::new(source).to_raw(),
            ))
        }
    }

    /// Creates a bfloat16 type.
    pub fn bfloat16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirBF16TypeGet(context.to_raw())) }
    }

    /// Creates a float16 type.
    pub fn float16(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF16TypeGet(context.to_raw())) }
    }

    /// Creates a float32 type.
    pub fn float32(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF32TypeGet(context.to_raw())) }
    }

    /// Creates a float64 type.
    pub fn float64(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirF64TypeGet(context.to_raw())) }
    }

    /// Creates a float4 E2M1FN type.
    pub fn float4_e2m1fn(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat4E2M1FNTypeGet(context.to_raw())) }
    }

    /// Creates a float6 E2M3FN type.
    pub fn float6_e2m3fn(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat6E2M3FNTypeGet(context.to_raw())) }
    }

    /// Creates a float6 E3M2FN type.
    pub fn float6_e3m2fn(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat6E3M2FNTypeGet(context.to_raw())) }
    }

    /// Creates a float8 E3M4 type.
    pub fn float8_e3m4(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E3M4TypeGet(context.to_raw())) }
    }

    /// Creates a float8 E4M3 type.
    pub fn float8_e4m3(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E4M3TypeGet(context.to_raw())) }
    }

    /// Creates a float8 E4M3B11FNUZ type.
    pub fn float8_e4m3b11fnuz(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E4M3B11FNUZTypeGet(context.to_raw())) }
    }

    /// Creates a float8 E4M3FN type.
    pub fn float8_e4m3fn(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E4M3FNTypeGet(context.to_raw())) }
    }

    /// Creates a float8 E4M3FNUZ type.
    pub fn float8_e4m3fnuz(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E4M3FNUZTypeGet(context.to_raw())) }
    }

    /// Creates a float8 E5M2 type.
    pub fn float8_e5m2(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E5M2TypeGet(context.to_raw())) }
    }

    /// Creates a float8 E5M2FNUZ type.
    pub fn float8_e5m2fnuz(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E5M2FNUZTypeGet(context.to_raw())) }
    }

    /// Creates a float8 E8M0FNU type.
    pub fn float8_e8m0fnu(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirFloat8E8M0FNUTypeGet(context.to_raw())) }
    }

    /// Returns the width of a float type. Only valid if `is_float()` is true.
    pub fn float_width(&self) -> u32 {
        unsafe { mlirFloatTypeGetWidth(self.raw) }
    }

    /// Creates an index type.
    pub fn index(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirIndexTypeGet(context.to_raw())) }
    }

    /// Creates a none type.
    pub fn none(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirNoneTypeGet(context.to_raw())) }
    }

    /// Creates a vector type.
    pub fn vector(dimensions: &[u64], r#type: Self) -> Self {
        unsafe {
            Self::from_raw(mlirVectorTypeGet(
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }

    /// Creates a vector type with diagnostics.
    pub fn vector_checked(
        location: Location<'c>,
        dimensions: &[u64],
        r#type: Self,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirVectorTypeGetChecked(
                location.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                r#type.raw,
            ))
        }
    }

    /// Creates a scalable vector type.
    pub fn vector_scalable(dimensions: &[u64], scalable: &[bool], element_type: Self) -> Self {
        unsafe {
            Self::from_raw(mlirVectorTypeGetScalable(
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                scalable.as_ptr(),
                element_type.raw,
            ))
        }
    }

    /// Creates a scalable vector type with diagnostics.
    pub fn vector_scalable_checked(
        location: Location<'c>,
        dimensions: &[u64],
        scalable: &[bool],
        element_type: Self,
    ) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirVectorTypeGetScalableChecked(
                location.to_raw(),
                dimensions.len() as isize,
                dimensions.as_ptr() as *const i64,
                scalable.as_ptr(),
                element_type.raw,
            ))
        }
    }

    /// Returns `true` if the vector type has at least one scalable dimension.
    pub fn is_scalable_vector(&self) -> bool {
        unsafe { mlirVectorTypeIsScalable(self.raw) }
    }

    /// Returns `true` if the given dimension of a vector type is scalable.
    pub fn is_vector_dim_scalable(&self, dim: usize) -> bool {
        unsafe { mlirVectorTypeIsDimScalable(self.raw, dim as isize) }
    }

    /// Creates a type from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirType) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Creates an optional type from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirType) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }
}

impl<'c> TypeLike<'c> for Type<'c> {
    fn to_raw(&self) -> MlirType {
        self.raw
    }
}

impl PartialEq for Type<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeEqual(self.raw, other.raw) }
    }
}

impl Eq for Type<'_> {}

impl Display for Type<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirTypePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Type<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "Type(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

impl std::hash::Hash for Type<'_> {
    // Hashes the type's pointer since they are unique w.r.t. the MLIR context.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.ptr.hash(state);
    }
}

from_subtypes!(
    Type,
    FunctionType,
    IntegerType,
    MemRefType,
    RankedTensorType,
    TupleType
);

#[cfg(test)]
mod tests {
    use crate::test::create_test_context;

    use super::*;

    #[test]
    fn new() {
        let context = create_test_context();
        Type::parse(&context, "f32");
    }

    #[test]
    fn integer() {
        let context = create_test_context();

        assert_eq!(
            Type::from(IntegerType::new(&context, 42)),
            Type::parse(&context, "i42").unwrap()
        );
    }

    #[test]
    fn index() {
        let context = create_test_context();

        assert_eq!(
            Type::index(&context),
            Type::parse(&context, "index").unwrap()
        );
    }

    #[test]
    fn vector() {
        let context = create_test_context();

        assert_eq!(
            Type::vector(&[42], Type::float64(&context)),
            Type::parse(&context, "vector<42xf64>").unwrap()
        );
    }

    #[test]
    #[ignore = "SIGABRT on llvm with assertions on"]
    fn vector_with_invalid_dimension() {
        let context = create_test_context();

        assert_eq!(
            Type::vector(&[0], IntegerType::new(&context, 32).into()).to_string(),
            "vector<0xi32>"
        );
    }

    #[test]
    fn vector_checked() {
        let context = create_test_context();

        assert_eq!(
            Type::vector_checked(
                Location::unknown(&context),
                &[42],
                IntegerType::new(&context, 32).into()
            ),
            Type::parse(&context, "vector<42xi32>")
        );
    }

    #[test]
    fn vector_checked_fail() {
        let context = create_test_context();

        assert_eq!(
            Type::vector_checked(Location::unknown(&context), &[0], Type::index(&context)),
            None
        );
    }

    #[test]
    fn equal() {
        let context = create_test_context();

        assert_eq!(Type::index(&context), Type::index(&context));
    }

    #[test]
    fn not_equal() {
        let context = create_test_context();

        assert_ne!(Type::index(&context), Type::float64(&context));
    }

    #[test]
    fn display() {
        let context = create_test_context();

        assert_eq!(Type::index(&context).to_string(), "index");
    }

    #[test]
    fn debug() {
        let context = create_test_context();

        assert_eq!(format!("{:?}", Type::index(&context)), "Type(index)");
    }

    #[test]
    fn float4_e2m1fn() {
        let context = create_test_context();

        assert_eq!(
            Type::float4_e2m1fn(&context),
            Type::parse(&context, "f4E2M1FN").unwrap()
        );
    }

    #[test]
    fn float6_e2m3fn() {
        let context = create_test_context();

        assert_eq!(
            Type::float6_e2m3fn(&context),
            Type::parse(&context, "f6E2M3FN").unwrap()
        );
    }

    #[test]
    fn float6_e3m2fn() {
        let context = create_test_context();

        assert_eq!(
            Type::float6_e3m2fn(&context),
            Type::parse(&context, "f6E3M2FN").unwrap()
        );
    }

    #[test]
    fn float8_e3m4() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e3m4(&context),
            Type::parse(&context, "f8E3M4").unwrap()
        );
    }

    #[test]
    fn float8_e4m3() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e4m3(&context),
            Type::parse(&context, "f8E4M3").unwrap()
        );
    }

    #[test]
    fn float8_e4m3b11fnuz() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e4m3b11fnuz(&context),
            Type::parse(&context, "f8E4M3B11FNUZ").unwrap()
        );
    }

    #[test]
    fn float8_e4m3fn() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e4m3fn(&context),
            Type::parse(&context, "f8E4M3FN").unwrap()
        );
    }

    #[test]
    fn float8_e4m3fnuz() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e4m3fnuz(&context),
            Type::parse(&context, "f8E4M3FNUZ").unwrap()
        );
    }

    #[test]
    fn float8_e5m2() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e5m2(&context),
            Type::parse(&context, "f8E5M2").unwrap()
        );
    }

    #[test]
    fn float8_e5m2fnuz() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e5m2fnuz(&context),
            Type::parse(&context, "f8E5M2FNUZ").unwrap()
        );
    }

    #[test]
    fn float8_e8m0fnu() {
        let context = create_test_context();

        assert_eq!(
            Type::float8_e8m0fnu(&context),
            Type::parse(&context, "f8E8M0FNU").unwrap()
        );
    }

    #[test]
    fn float_width() {
        let context = create_test_context();

        assert_eq!(Type::float32(&context).float_width(), 32);
        assert_eq!(Type::float64(&context).float_width(), 64);
        assert_eq!(Type::float16(&context).float_width(), 16);
    }

    #[test]
    fn vector_scalable() {
        let context = create_test_context();

        assert_eq!(
            Type::vector_scalable(&[4], &[true], Type::float32(&context)),
            Type::parse(&context, "vector<[4]xf32>").unwrap()
        );
    }

    #[test]
    fn vector_scalable_checked() {
        let context = create_test_context();

        assert_eq!(
            Type::vector_scalable_checked(
                Location::unknown(&context),
                &[4],
                &[true],
                Type::float32(&context),
            ),
            Type::parse(&context, "vector<[4]xf32>")
        );
    }

    #[test]
    fn vector_scalable_checked_fail() {
        let context = create_test_context();

        assert_eq!(
            Type::vector_scalable_checked(
                Location::unknown(&context),
                &[0],
                &[true],
                Type::index(&context),
            ),
            None
        );
    }

    #[test]
    fn is_scalable_vector() {
        let context = create_test_context();

        assert!(Type::vector_scalable(&[4], &[true], Type::float32(&context)).is_scalable_vector());
        assert!(!Type::vector(&[4], Type::float32(&context)).is_scalable_vector());
    }

    #[test]
    fn is_vector_dim_scalable() {
        let context = create_test_context();

        let scalable = Type::vector_scalable(&[4, 8], &[true, false], Type::float32(&context));
        assert!(scalable.is_vector_dim_scalable(0));
        assert!(!scalable.is_vector_dim_scalable(1));
    }
}
