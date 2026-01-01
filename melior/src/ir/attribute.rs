//! Attributes.

#[macro_use]
mod r#macro;
mod array;
mod attribute_like;
mod bool;
mod dense_elements;
mod dense_i32_array;
mod dense_i64_array;
mod distinct;
mod flat_symbol_ref;
mod float;
mod integer;
mod string;
mod r#type;

pub use self::{
    array::ArrayAttribute, attribute_like::AttributeLike, bool::BoolAttribute,
    dense_elements::DenseElementsAttribute, dense_i32_array::DenseI32ArrayAttribute,
    dense_i64_array::DenseI64ArrayAttribute, flat_symbol_ref::FlatSymbolRefAttribute,
    float::FloatAttribute, integer::IntegerAttribute, r#type::TypeAttribute,
    string::StringAttribute,
};
use crate::{context::Context, string_ref::StringRef, utility::print_callback};
use distinct::DisctinctAttribute;
use mlir_sys::{
    mlirAttributeEqual, mlirAttributeGetNull, mlirAttributeParseGet, mlirAttributePrint,
    mlirUnitAttrGet, MlirAttribute,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An attribute.
// Attributes are always values but their internal storage is owned by contexts.
#[derive(Clone, Copy)]
pub struct Attribute<'c> {
    raw: MlirAttribute,
    _context: PhantomData<&'c Context>,
}

impl<'c> Attribute<'c> {
    /// Parses an attribute.
    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        unsafe {
            Self::from_option_raw(mlirAttributeParseGet(
                context.to_raw(),
                StringRef::new(source).to_raw(),
            ))
        }
    }

    /// Creates a unit attribute.
    pub fn unit(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirUnitAttrGet(context.to_raw())) }
    }

    pub(crate) unsafe fn null() -> Self {
        unsafe { Self::from_raw(mlirAttributeGetNull()) }
    }

    /// Creates an attribute from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Creates an optional attribute from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirAttribute) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> AttributeLike<'c> for Attribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.raw
    }
}

impl PartialEq for Attribute<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAttributeEqual(self.raw, other.raw) }
    }
}

impl Eq for Attribute<'_> {}

impl Display for Attribute<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirAttributePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Attribute<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

impl std::hash::Hash for Attribute<'_> {
    // Hashes the attribute's pointer since they are unique w.r.t. the MLIR context.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.ptr.hash(state);
    }
}

from_subtypes!(
    Attribute,
    ArrayAttribute,
    BoolAttribute,
    DenseElementsAttribute,
    DenseI32ArrayAttribute,
    DenseI64ArrayAttribute,
    FlatSymbolRefAttribute,
    FloatAttribute,
    IntegerAttribute,
    StringAttribute,
    TypeAttribute,
    DisctinctAttribute,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Type, TypeLike},
        test::create_test_context,
    };

    #[test]
    fn parse() {
        let context = create_test_context();
        for attribute in ["unit", "i32", r#""foo""#] {
            assert!(Attribute::parse(&context, attribute).is_some());
        }
    }

    #[test]
    fn parse_none() {
        // Note: this test will print a warning if LLVM was compiled with asserts.
        // `<mlir_parser_buffer>:1:1: error: expected attribute value
        // z
        // ^`
        assert!(Attribute::parse(&Context::new(), "z").is_none());
    }

    #[test]
    fn context() {
        let context = create_test_context();
        Attribute::parse(&context, "unit").unwrap().context();
    }

    #[test]
    fn r#type() {
        let context = Context::new();

        assert_eq!(
            Attribute::parse(&context, "unit").unwrap().r#type(),
            Type::none(&context)
        );
    }

    // TODO Fix this.
    #[ignore]
    #[test]
    fn type_id() {
        let context = Context::new();

        assert_eq!(
            Attribute::parse(&context, "42 : index").unwrap().type_id(),
            Type::index(&context).id()
        );
    }

    #[test]
    fn is_array() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "[]").unwrap().is_array());
    }

    #[test]
    fn is_bool() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "false").unwrap().is_bool());
    }

    #[test]
    fn is_dense_elements() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "dense<10> : tensor<2xi8>")
            .unwrap()
            .is_dense_elements());
    }

    #[test]
    fn is_dense_int_elements() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "dense<42> : tensor<42xi8>")
            .unwrap()
            .is_dense_int_elements());
    }

    #[test]
    fn is_dense_fp_elements() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "dense<42.0> : tensor<42xf32>")
            .unwrap()
            .is_dense_fp_elements());
    }

    #[test]
    fn is_elements() {
        let context = create_test_context();
        assert!(Attribute::parse(
            &context,
            "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>"
        )
        .unwrap()
        .is_elements());
    }

    #[test]
    fn is_integer() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "42").unwrap().is_integer());
    }

    #[test]
    fn is_integer_set() {
        let context = create_test_context();
        assert!(
            Attribute::parse(&context, "affine_set<(d0) : (d0 - 2 >= 0)>")
                .unwrap()
                .is_integer_set()
        );
    }

    // TODO Fix this.
    #[ignore]
    #[test]
    fn is_opaque() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "#foo<\"bar\">")
            .unwrap()
            .is_opaque());
    }

    #[test]
    fn is_sparse_elements() {
        let context = create_test_context();
        assert!(Attribute::parse(
            &context,
            "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>"
        )
        .unwrap()
        .is_sparse_elements());
    }

    #[test]
    fn is_string() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "\"foo\"").unwrap().is_string());
    }

    #[test]
    fn is_type() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "index").unwrap().is_type());
    }

    #[test]
    fn is_unit() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "unit").unwrap().is_unit());
    }

    #[test]
    fn is_symbol() {
        let context = create_test_context();
        assert!(Attribute::parse(&context, "@foo").unwrap().is_symbol_ref());
    }

    #[test]
    fn equal() {
        let context = create_test_context();
        let attribute = Attribute::parse(&context, "unit").unwrap();

        assert_eq!(attribute, attribute);
    }

    #[test]
    fn not_equal() {
        let context = create_test_context();

        assert_ne!(
            Attribute::parse(&context, "unit").unwrap(),
            Attribute::parse(&context, "42").unwrap()
        );
    }

    #[test]
    fn display() {
        let context = create_test_context();
        assert_eq!(
            Attribute::parse(&context, "unit").unwrap().to_string(),
            "unit"
        );
    }
}
