use super::{ArrayAttribute, Attribute, AttributeLike};
use crate::{
    Error, StringRef,
    ir::{Type, TypeLike, r#type::IntegerType},
};
use mlir_sys::{
    MlirAttribute, MlirStringRef, mlirComplexTypeGetElementType, mlirDenseElementsAttrBoolGet,
    mlirDenseElementsAttrBoolSplatGet, mlirDenseElementsAttrDoubleGet,
    mlirDenseElementsAttrDoubleSplatGet, mlirDenseElementsAttrFloatGet,
    mlirDenseElementsAttrFloatSplatGet, mlirDenseElementsAttrGet,
    mlirDenseElementsAttrGetBoolSplatValue, mlirDenseElementsAttrGetBoolValue,
    mlirDenseElementsAttrGetDoubleSplatValue, mlirDenseElementsAttrGetDoubleValue,
    mlirDenseElementsAttrGetFloatSplatValue, mlirDenseElementsAttrGetFloatValue,
    mlirDenseElementsAttrGetIndexValue, mlirDenseElementsAttrGetInt8SplatValue,
    mlirDenseElementsAttrGetInt8Value, mlirDenseElementsAttrGetInt16Value,
    mlirDenseElementsAttrGetInt32SplatValue, mlirDenseElementsAttrGetInt32Value,
    mlirDenseElementsAttrGetInt64SplatValue, mlirDenseElementsAttrGetInt64Value,
    mlirDenseElementsAttrGetRawData, mlirDenseElementsAttrGetSplatValue,
    mlirDenseElementsAttrGetStringSplatValue, mlirDenseElementsAttrGetStringValue,
    mlirDenseElementsAttrGetUInt8Value, mlirDenseElementsAttrGetUInt16Value,
    mlirDenseElementsAttrGetUInt32SplatValue, mlirDenseElementsAttrGetUInt32Value,
    mlirDenseElementsAttrGetUInt64SplatValue, mlirDenseElementsAttrGetUInt64Value,
    mlirDenseElementsAttrInt8Get, mlirDenseElementsAttrInt8SplatGet, mlirDenseElementsAttrInt16Get,
    mlirDenseElementsAttrInt32Get, mlirDenseElementsAttrInt32SplatGet,
    mlirDenseElementsAttrInt64Get, mlirDenseElementsAttrInt64SplatGet,
    mlirDenseElementsAttrIsSplat, mlirDenseElementsAttrRawBufferGet,
    mlirDenseElementsAttrReshapeGet, mlirDenseElementsAttrSplatGet, mlirDenseElementsAttrStringGet,
    mlirDenseElementsAttrUInt8Get, mlirDenseElementsAttrUInt16Get, mlirDenseElementsAttrUInt32Get,
    mlirDenseElementsAttrUInt32SplatGet, mlirDenseElementsAttrUInt64Get,
    mlirDenseElementsAttrUInt64SplatGet, mlirElementsAttrGetNumElements, mlirShapedTypeGetDimSize,
    mlirShapedTypeGetElementType, mlirShapedTypeGetRank, mlirShapedTypeHasStaticShape,
};

macro_rules! dense_elements_constructor {
    ($name:ident, $type:ty, $element:expr, $function:ident) => {
        #[doc = concat!(" Creates a dense elements attribute from ", stringify!($type), " values.")]
        pub fn $name(r#type: Type<'c>, values: &[$type]) -> Result<Self, Error> {
            Self::check_values(r#type, $element, stringify!($type), values.len())?;

            Ok(unsafe {
                Self::from_raw($function(
                    r#type.to_raw(),
                    values.len() as isize,
                    values.as_ptr(),
                ))
            })
        }
    };
}

macro_rules! dense_elements_splat_constructor {
    ($name:ident, $type:ty, $element:expr, $function:ident) => {
        #[doc = concat!(" Creates a splat dense elements attribute with the given ", stringify!($type), " value.")]
        pub fn $name(r#type: Type<'c>, value: $type) -> Result<Self, Error> {
            Self::check_values(r#type, $element, stringify!($type), 1)?;

            Ok(unsafe { Self::from_raw($function(r#type.to_raw(), value)) })
        }
    };
}

macro_rules! dense_element_accessor {
    ($name:ident, $type:ty, $element:expr, $type_name:expr, $ffi:ident) => {
        #[doc = concat!(" Returns an element of `", stringify!($type), "` at the given index.")]
        pub fn $name(&self, index: usize) -> Result<$type, Error> {
            if !self.has_elements($element) {
                Err(Error::ElementExpected {
                    r#type: $type_name,
                    value: self.to_string(),
                })
            } else if index < self.len() {
                Ok(unsafe { $ffi(self.attribute.to_raw(), index as isize) })
            } else {
                Err(Error::PositionOutOfBounds {
                    name: "dense element",
                    value: self.to_string(),
                    index,
                })
            }
        }
    };
}

macro_rules! dense_elements_splat_accessor {
    ($name:ident, $type:ty, $element:expr, $function:ident) => {
        #[doc = concat!(" Returns the ", stringify!($type), " splat value.")]
        pub fn $name(&self) -> Result<$type, Error> {
            self.check_splat($element, stringify!($type))?;

            Ok(unsafe { $function(self.attribute.to_raw()) })
        }
    };
}

#[derive(Clone, Copy)]
enum Element {
    Bool,
    Integer { width: u32, signed: bool },
    Float { width: u32 },
    String,
}

impl Element {
    const I8: Self = Self::Integer {
        width: 8,
        signed: true,
    };
    const I16: Self = Self::Integer {
        width: 16,
        signed: true,
    };
    const I32: Self = Self::Integer {
        width: 32,
        signed: true,
    };
    const I64: Self = Self::Integer {
        width: 64,
        signed: true,
    };
    const U8: Self = Self::Integer {
        width: 8,
        signed: false,
    };
    const U16: Self = Self::Integer {
        width: 16,
        signed: false,
    };
    const U32: Self = Self::Integer {
        width: 32,
        signed: false,
    };
    const U64: Self = Self::Integer {
        width: 64,
        signed: false,
    };
    const F32: Self = Self::Float { width: 32 };
    const F64: Self = Self::Float { width: 64 };

    fn matches(self, r#type: Type) -> bool {
        match self {
            Self::Bool => IntegerType::try_from(r#type).is_ok_and(|r#type| r#type.width() == 1),
            Self::Integer { width, signed } => {
                if r#type.is_index() {
                    width == 64
                } else {
                    IntegerType::try_from(r#type).is_ok_and(|r#type| {
                        r#type.width() == width
                            && (r#type.is_signless() || r#type.is_signed() == signed)
                    })
                }
            }
            Self::Float { width } => r#type.is_float() && r#type.float_width() == width,
            Self::String => {
                !(r#type.is_integer()
                    || r#type.is_index()
                    || r#type.is_float()
                    || r#type.is_complex())
            }
        }
    }
}

/// A dense elements attribute.
#[derive(Clone, Copy, Hash)]
pub struct DenseElementsAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DenseElementsAttribute<'c> {
    /// Creates a dense elements attribute.
    pub fn new(r#type: Type<'c>, values: &[Attribute<'c>]) -> Result<Self, Error> {
        Self::check_attributes(r#type, values)?;

        Ok(unsafe {
            Self::from_raw(mlirDenseElementsAttrGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr() as *const _ as *const _,
            ))
        })
    }

    /// Creates a dense elements attribute from bool values.
    ///
    /// The C API takes `int*`, so values are passed as `&[i32]` where non-zero
    /// means true.
    pub fn bool_values(r#type: Type<'c>, values: &[i32]) -> Result<Self, Error> {
        Self::check_values(r#type, Element::Bool, "bool", values.len())?;

        Ok(unsafe {
            Self::from_raw(mlirDenseElementsAttrBoolGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        })
    }

    dense_elements_constructor!(i8_values, i8, Element::I8, mlirDenseElementsAttrInt8Get);
    dense_elements_constructor!(i16_values, i16, Element::I16, mlirDenseElementsAttrInt16Get);
    dense_elements_constructor!(i32_values, i32, Element::I32, mlirDenseElementsAttrInt32Get);
    dense_elements_constructor!(i64_values, i64, Element::I64, mlirDenseElementsAttrInt64Get);
    dense_elements_constructor!(u8_values, u8, Element::U8, mlirDenseElementsAttrUInt8Get);
    dense_elements_constructor!(
        u16_values,
        u16,
        Element::U16,
        mlirDenseElementsAttrUInt16Get
    );
    dense_elements_constructor!(
        u32_values,
        u32,
        Element::U32,
        mlirDenseElementsAttrUInt32Get
    );
    dense_elements_constructor!(
        u64_values,
        u64,
        Element::U64,
        mlirDenseElementsAttrUInt64Get
    );
    dense_elements_constructor!(f32_values, f32, Element::F32, mlirDenseElementsAttrFloatGet);
    dense_elements_constructor!(
        f64_values,
        f64,
        Element::F64,
        mlirDenseElementsAttrDoubleGet
    );

    /// Creates a dense elements attribute from string values.
    pub fn string_values(r#type: Type<'c>, values: &[StringRef]) -> Result<Self, Error> {
        Self::check_values(r#type, Element::String, "string", values.len())?;

        let mut raws: Vec<MlirStringRef> = values.iter().map(|s| s.to_raw()).collect();

        Ok(unsafe {
            Self::from_raw(mlirDenseElementsAttrStringGet(
                r#type.to_raw(),
                raws.len() as isize,
                raws.as_mut_ptr(),
            ))
        })
    }

    /// Creates a dense elements attribute from a raw buffer of packed element
    /// data, or returns `None` if the buffer does not match the type.
    pub fn raw_buffer(r#type: Type<'c>, buffer: &[u8]) -> Option<Self> {
        if Self::element_count(r#type).is_err()
            || Element::String.matches(Self::shaped_element_type(r#type))
        {
            return None;
        }

        unsafe {
            Attribute::from_option_raw(mlirDenseElementsAttrRawBufferGet(
                r#type.to_raw(),
                buffer.len(),
                buffer.as_ptr() as *const _,
            ))
        }
        .map(|attribute| Self { attribute })
    }

    // -------------------------------------------------------------------------
    // Splat constructors
    // -------------------------------------------------------------------------

    /// Creates a splat dense elements attribute with the given element
    /// attribute.
    pub fn splat(r#type: Type<'c>, element: Attribute<'c>) -> Result<Self, Error> {
        Self::check_attributes(r#type, &[element])?;

        Ok(unsafe {
            Self::from_raw(mlirDenseElementsAttrSplatGet(
                r#type.to_raw(),
                element.to_raw(),
            ))
        })
    }

    dense_elements_splat_constructor!(
        bool_splat,
        bool,
        Element::Bool,
        mlirDenseElementsAttrBoolSplatGet
    );
    dense_elements_splat_constructor!(i8_splat, i8, Element::I8, mlirDenseElementsAttrInt8SplatGet);
    dense_elements_splat_constructor!(
        i32_splat,
        i32,
        Element::I32,
        mlirDenseElementsAttrInt32SplatGet
    );
    dense_elements_splat_constructor!(
        i64_splat,
        i64,
        Element::I64,
        mlirDenseElementsAttrInt64SplatGet
    );
    dense_elements_splat_constructor!(
        u32_splat,
        u32,
        Element::U32,
        mlirDenseElementsAttrUInt32SplatGet
    );
    dense_elements_splat_constructor!(
        u64_splat,
        u64,
        Element::U64,
        mlirDenseElementsAttrUInt64SplatGet
    );
    dense_elements_splat_constructor!(
        f32_splat,
        f32,
        Element::F32,
        mlirDenseElementsAttrFloatSplatGet
    );
    dense_elements_splat_constructor!(
        f64_splat,
        f64,
        Element::F64,
        mlirDenseElementsAttrDoubleSplatGet
    );

    // -------------------------------------------------------------------------
    // Misc
    // -------------------------------------------------------------------------

    /// Returns a length.
    pub fn len(&self) -> usize {
        (unsafe { mlirElementsAttrGetNumElements(self.attribute.to_raw()) }) as usize
    }

    /// Checks if an array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks whether this dense elements attribute is a splat.
    pub fn is_splat(&self) -> bool {
        unsafe { mlirDenseElementsAttrIsSplat(self.attribute.to_raw()) }
    }

    /// Returns a reshaped dense elements attribute with the given shaped type.
    ///
    /// The new type must have the same total number of elements.
    pub fn reshape(&self, r#type: Type<'c>) -> Result<Self, Error> {
        let count = Self::element_count(r#type)?;

        if self.has_elements(Element::String) {
            Err(Error::AttributeExpected(
                "numeric dense elements",
                self.to_string(),
            ))
        } else if Self::shaped_element_type(r#type) != self.element_type() {
            Err(Error::ElementExpected {
                r#type: "the same",
                value: r#type.to_string(),
            })
        } else if count == self.len() {
            Ok(unsafe {
                Self::from_raw(mlirDenseElementsAttrReshapeGet(
                    self.attribute.to_raw(),
                    r#type.to_raw(),
                ))
            })
        } else {
            Err(Error::CountMismatch {
                name: "dense element",
                expected: self.len(),
                actual: count,
            })
        }
    }

    /// Returns a raw pointer to the packed element data.
    pub fn raw_data(&self) -> Result<*const core::ffi::c_void, Error> {
        if self.has_elements(Element::String) {
            Err(Error::AttributeExpected(
                "numeric dense elements",
                self.to_string(),
            ))
        } else {
            Ok(unsafe { mlirDenseElementsAttrGetRawData(self.attribute.to_raw()) })
        }
    }

    // -------------------------------------------------------------------------
    // Element accessors
    // -------------------------------------------------------------------------

    dense_element_accessor!(
        bool_element,
        bool,
        Element::Bool,
        "bool",
        mlirDenseElementsAttrGetBoolValue
    );

    dense_element_accessor!(
        i8_element,
        i8,
        Element::I8,
        "i8",
        mlirDenseElementsAttrGetInt8Value
    );

    dense_element_accessor!(
        i16_element,
        i16,
        Element::I16,
        "i16",
        mlirDenseElementsAttrGetInt16Value
    );

    dense_element_accessor!(
        i32_element,
        i32,
        Element::I32,
        "i32",
        mlirDenseElementsAttrGetInt32Value
    );

    dense_element_accessor!(
        i64_element,
        i64,
        Element::I64,
        "i64",
        mlirDenseElementsAttrGetInt64Value
    );

    dense_element_accessor!(
        u8_element,
        u8,
        Element::U8,
        "u8",
        mlirDenseElementsAttrGetUInt8Value
    );

    dense_element_accessor!(
        u16_element,
        u16,
        Element::U16,
        "u16",
        mlirDenseElementsAttrGetUInt16Value
    );

    dense_element_accessor!(
        u32_element,
        u32,
        Element::U32,
        "u32",
        mlirDenseElementsAttrGetUInt32Value
    );

    dense_element_accessor!(
        u64_element,
        u64,
        Element::U64,
        "u64",
        mlirDenseElementsAttrGetUInt64Value
    );

    dense_element_accessor!(
        f32_element,
        f32,
        Element::F32,
        "f32",
        mlirDenseElementsAttrGetFloatValue
    );

    dense_element_accessor!(
        f64_element,
        f64,
        Element::F64,
        "f64",
        mlirDenseElementsAttrGetDoubleValue
    );

    dense_element_accessor!(
        index_element,
        u64,
        Element::U64,
        "index",
        mlirDenseElementsAttrGetIndexValue
    );

    /// Returns a string element at the given index.
    pub fn string_element(&self, index: usize) -> Result<&str, Error> {
        if !self.has_elements(Element::String) {
            Err(Error::ElementExpected {
                r#type: "string",
                value: self.to_string(),
            })
        } else if index < self.len() {
            let raw = unsafe {
                mlirDenseElementsAttrGetStringValue(self.attribute.to_raw(), index as isize)
            };

            unsafe { StringRef::from_raw(raw) }
                .as_str()
                .map_err(|_| Error::ElementExpected {
                    r#type: "UTF-8 string",
                    value: self.to_string(),
                })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    // -------------------------------------------------------------------------
    // Splat value accessors
    // -------------------------------------------------------------------------

    /// Returns the splat value as a generic attribute.
    pub fn splat_value(&self) -> Result<Attribute<'c>, Error> {
        if self.is_splat() {
            Ok(unsafe {
                Attribute::from_raw(mlirDenseElementsAttrGetSplatValue(self.attribute.to_raw()))
            })
        } else {
            Err(Error::AttributeExpected(
                "splat dense elements",
                self.to_string(),
            ))
        }
    }

    /// Returns the bool splat value. Non-zero means true.
    pub fn bool_splat_value(&self) -> Result<bool, Error> {
        self.check_splat(Element::Bool, "bool")?;

        Ok(unsafe { mlirDenseElementsAttrGetBoolSplatValue(self.attribute.to_raw()) != 0 })
    }

    dense_elements_splat_accessor!(
        i8_splat_value,
        i8,
        Element::I8,
        mlirDenseElementsAttrGetInt8SplatValue
    );
    dense_elements_splat_accessor!(
        i32_splat_value,
        i32,
        Element::I32,
        mlirDenseElementsAttrGetInt32SplatValue
    );
    dense_elements_splat_accessor!(
        i64_splat_value,
        i64,
        Element::I64,
        mlirDenseElementsAttrGetInt64SplatValue
    );
    dense_elements_splat_accessor!(
        u32_splat_value,
        u32,
        Element::U32,
        mlirDenseElementsAttrGetUInt32SplatValue
    );
    dense_elements_splat_accessor!(
        u64_splat_value,
        u64,
        Element::U64,
        mlirDenseElementsAttrGetUInt64SplatValue
    );
    dense_elements_splat_accessor!(
        f32_splat_value,
        f32,
        Element::F32,
        mlirDenseElementsAttrGetFloatSplatValue
    );
    dense_elements_splat_accessor!(
        f64_splat_value,
        f64,
        Element::F64,
        mlirDenseElementsAttrGetDoubleSplatValue
    );

    /// Returns the string splat value.
    pub fn string_splat_value(&self) -> Result<&str, Error> {
        self.check_splat(Element::String, "string")?;

        let raw = unsafe { mlirDenseElementsAttrGetStringSplatValue(self.attribute.to_raw()) };

        unsafe { StringRef::from_raw(raw) }
            .as_str()
            .map_err(|_| Error::ElementExpected {
                r#type: "UTF-8 string",
                value: self.to_string(),
            })
    }

    fn element_type(&self) -> Type<'c> {
        Self::shaped_element_type(self.attribute.r#type())
    }

    fn has_elements(&self, element: Element) -> bool {
        element.matches(self.element_type())
            && match element {
                Element::Bool | Element::Integer { .. } => self.is_dense_int_elements(),
                Element::Float { .. } => self.is_dense_fp_elements(),
                Element::String => true,
            }
    }

    fn check_splat(&self, element: Element, name: &'static str) -> Result<(), Error> {
        if !self.has_elements(element) {
            return Err(Error::ElementExpected {
                r#type: name,
                value: self.to_string(),
            });
        }

        if !self.is_splat() {
            return Err(Error::AttributeExpected(
                "splat dense elements",
                self.to_string(),
            ));
        }

        Ok(())
    }

    fn element_count(r#type: Type<'c>) -> Result<usize, Error> {
        if r#type.is_shaped() && unsafe { mlirShapedTypeHasStaticShape(r#type.to_raw()) } {
            Ok((0..unsafe { mlirShapedTypeGetRank(r#type.to_raw()) })
                .map(
                    |index| unsafe { mlirShapedTypeGetDimSize(r#type.to_raw(), index as isize) }
                        as usize,
                )
                .product())
        } else {
            Err(Error::TypeExpected("statically shaped", r#type.to_string()))
        }
    }

    fn shaped_element_type(r#type: Type<'c>) -> Type<'c> {
        unsafe { Type::from_raw(mlirShapedTypeGetElementType(r#type.to_raw())) }
    }

    const fn check_count(count: usize, len: usize) -> Result<(), Error> {
        if len != 1 && len != count {
            return Err(Error::CountMismatch {
                name: "dense element",
                expected: count,
                actual: len,
            });
        }

        Ok(())
    }

    fn check_values(
        r#type: Type<'c>,
        element: Element,
        name: &'static str,
        len: usize,
    ) -> Result<(), Error> {
        let count = Self::element_count(r#type)?;

        if !element.matches(Self::shaped_element_type(r#type)) {
            return Err(Error::ElementExpected {
                r#type: name,
                value: r#type.to_string(),
            });
        }

        Self::check_count(count, len)
    }

    fn check_attributes(r#type: Type<'c>, values: &[Attribute<'c>]) -> Result<(), Error> {
        Self::check_count(Self::element_count(r#type)?, values.len())?;

        let element_type = Self::shaped_element_type(r#type);

        if let Some(value) = values
            .iter()
            .find(|value| !Self::is_element(element_type, **value))
        {
            return Err(Error::AttributeExpected("dense element", value.to_string()));
        }

        Ok(())
    }

    fn is_element(r#type: Type<'c>, value: Attribute<'c>) -> bool {
        if r#type.is_integer() || r#type.is_index() {
            value.is_integer() && value.r#type() == r#type
        } else if r#type.is_float() {
            value.is_float() && value.r#type() == r#type
        } else if r#type.is_complex() {
            let r#type = unsafe { Type::from_raw(mlirComplexTypeGetElementType(r#type.to_raw())) };

            ArrayAttribute::try_from(value).is_ok_and(|array| {
                array.len() == 2
                    && (0..2).all(|index| {
                        array
                            .element(index)
                            .is_ok_and(|value| Self::is_element(r#type, value))
                    })
            })
        } else {
            value.is_string()
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
            r#type::{IntegerType, MemRefType, RankedTensorType},
        },
        test::create_test_context,
    };

    fn i32_tensor_type<'c>(context: &'c crate::Context, len: u64) -> Type<'c> {
        let element = IntegerType::new(context, 32).into();
        RankedTensorType::new(&[len], element, None).into()
    }

    fn i64_tensor_type<'c>(context: &'c crate::Context, len: u64) -> Type<'c> {
        let element = IntegerType::new(context, 64).into();
        RankedTensorType::new(&[len], element, None).into()
    }

    fn f32_tensor_type<'c>(context: &'c crate::Context, len: u64) -> Type<'c> {
        let element = Type::float32(context);
        RankedTensorType::new(&[len], element, None).into()
    }

    fn f64_tensor_type<'c>(context: &'c crate::Context, len: u64) -> Type<'c> {
        let element = Type::float64(context);
        RankedTensorType::new(&[len], element, None).into()
    }

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

    #[test]
    fn i32_values_and_elements() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i32_values(i32_tensor_type(&context, 3), &[10, 20, 30])
            .unwrap();
        assert_eq!(attr.i32_element(0), Ok(10));
        assert_eq!(attr.i32_element(1), Ok(20));
        assert_eq!(attr.i32_element(2), Ok(30));
    }

    #[test]
    fn i64_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::i64_values(i64_tensor_type(&context, 2), &[100i64, 200i64])
                .unwrap();
        assert_eq!(attr.i64_element(0), Ok(100));
        assert_eq!(attr.i64_element(1), Ok(200));
    }

    #[test]
    fn i8_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::i8_values(ty, &[1i8, -1i8]).unwrap();
        assert_eq!(attr.i8_element(0), Ok(1));
        assert_eq!(attr.i8_element(1), Ok(-1));
    }

    #[test]
    fn i16_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 16).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::i16_values(ty, &[300i16, -300i16]).unwrap();
        assert_eq!(attr.i16_element(0), Ok(300));
        assert_eq!(attr.i16_element(1), Ok(-300));
    }

    #[test]
    fn u8_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        let attr = DenseElementsAttribute::u8_values(ty, &[0u8, 128u8, 255u8]).unwrap();
        assert_eq!(attr.u8_element(0), Ok(0));
        assert_eq!(attr.u8_element(1), Ok(128));
        assert_eq!(attr.u8_element(2), Ok(255));
    }

    #[test]
    fn u16_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 16).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u16_values(ty, &[1000u16, 2000u16]).unwrap();
        assert_eq!(attr.u16_element(0), Ok(1000));
        assert_eq!(attr.u16_element(1), Ok(2000));
    }

    #[test]
    fn u32_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u32_values(ty, &[7u32, 42u32]).unwrap();
        assert_eq!(attr.u32_element(0), Ok(7));
        assert_eq!(attr.u32_element(1), Ok(42));
    }

    #[test]
    fn u64_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 64).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u64_values(ty, &[999u64, 1000u64]).unwrap();
        assert_eq!(attr.u64_element(0), Ok(999));
        assert_eq!(attr.u64_element(1), Ok(1000));
    }

    #[test]
    fn f32_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f32_values(f32_tensor_type(&context, 2), &[1.5f32, -2.5f32])
                .unwrap();
        assert_eq!(attr.f32_element(0), Ok(1.5f32));
        assert_eq!(attr.f32_element(1), Ok(-2.5f32));
    }

    #[test]
    fn f64_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f64_values(f64_tensor_type(&context, 2), &[1.25f64, -2.72f64])
                .unwrap();
        assert_eq!(attr.f64_element(0), Ok(1.25f64));
        assert_eq!(attr.f64_element(1), Ok(-2.72f64));
    }

    #[test]
    fn bool_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 1).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        // Non-zero = true, 0 = false
        let attr = DenseElementsAttribute::bool_values(ty, &[1, 0, 1]).unwrap();
        assert_eq!(attr.bool_element(0), Ok(true));
        assert_eq!(attr.bool_element(1), Ok(false));
        assert_eq!(attr.bool_element(2), Ok(true));
    }

    #[test]
    fn string_values_and_elements() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let element = Type::parse(&context, "!foo.string").unwrap();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let strings = [StringRef::new("hello"), StringRef::new("world")];
        let attr = DenseElementsAttribute::string_values(ty, &strings).unwrap();
        assert_eq!(attr.string_element(0), Ok("hello"));
        assert_eq!(attr.string_element(1), Ok("world"));
    }

    #[test]
    fn splat_i32() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i32_splat(i32_tensor_type(&context, 4), 77).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.i32_splat_value(), Ok(77));
        assert_eq!(attr.i32_element(0), Ok(77));
        assert_eq!(attr.i32_element(3), Ok(77));
    }

    #[test]
    fn splat_i64() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i64_splat(i64_tensor_type(&context, 3), 999i64).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.i64_splat_value(), Ok(999));
    }

    #[test]
    fn splat_u32() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let ty = RankedTensorType::new(&[5], element, None).into();
        let attr = DenseElementsAttribute::u32_splat(ty, 42u32).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.u32_splat_value(), Ok(42));
    }

    #[test]
    fn splat_u64() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 64).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u64_splat(ty, 12345u64).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.u64_splat_value(), Ok(12345));
    }

    #[test]
    fn splat_f32() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f32_splat(f32_tensor_type(&context, 4), 1.25f32).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.f32_splat_value(), Ok(1.25f32));
    }

    #[test]
    fn splat_f64() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f64_splat(f64_tensor_type(&context, 2), 2.72f64).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.f64_splat_value(), Ok(2.72f64));
    }

    #[test]
    fn splat_bool() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 1).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        let attr = DenseElementsAttribute::bool_splat(ty, true).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.bool_splat_value(), Ok(true));
    }

    #[test]
    fn splat_i8() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[4], element, None).into();
        let attr = DenseElementsAttribute::i8_splat(ty, -5i8).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.i8_splat_value(), Ok(-5));
    }

    #[test]
    fn splat_attribute() {
        let context = create_test_context();
        let integer_type = IntegerType::new(&context, 32);
        let ty = i32_tensor_type(&context, 3);
        let elem = IntegerAttribute::new(integer_type.into(), 55).into();
        let attr = DenseElementsAttribute::splat(ty, elem).unwrap();
        assert!(attr.is_splat());
        assert_eq!(attr.i32_element(0), Ok(55));
    }

    #[test]
    fn raw_buffer_round_trip() {
        let context = create_test_context();
        // Build a tensor of 4 x i32 from a raw byte buffer
        let values: [i32; 4] = [1, 2, 3, 4];
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
        let ty = i32_tensor_type(&context, 4);
        let attr = DenseElementsAttribute::raw_buffer(ty, bytes).unwrap();
        assert_eq!(attr.len(), 4);
        assert!(!attr.raw_data().unwrap().is_null());
    }

    #[test]
    fn reshape() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let original_ty = RankedTensorType::new(&[6], element, None).into();
        let attr = DenseElementsAttribute::i32_values(original_ty, &[1, 2, 3, 4, 5, 6]).unwrap();
        let new_ty = RankedTensorType::new(&[2, 3], element, None).into();
        let reshaped = attr.reshape(new_ty).unwrap();
        assert_eq!(reshaped.len(), 6);
        assert_eq!(reshaped.i32_element(0), Ok(1));
        assert_eq!(reshaped.i32_element(5), Ok(6));
    }

    #[test]
    fn out_of_bounds_error() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap();
        assert!(matches!(
            attr.i32_element(5),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn wrong_type_error() {
        let context = create_test_context();
        // Create a float tensor and try accessing as int
        let attr =
            DenseElementsAttribute::f32_values(f32_tensor_type(&context, 2), &[1.0, 2.0]).unwrap();
        assert!(matches!(
            attr.i32_element(0),
            Err(Error::ElementExpected { .. })
        ));
        // Create an int tensor and try accessing as float
        let attr2 =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap();
        assert!(matches!(
            attr2.f32_element(0),
            Err(Error::ElementExpected { .. })
        ));
    }

    #[test]
    fn fail_to_create_values_with_wrong_count() {
        let context = create_test_context();

        assert_eq!(
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 64), &[1, 2]).unwrap_err(),
            Error::CountMismatch {
                name: "dense element",
                expected: 64,
                actual: 2
            }
        );
    }

    #[test]
    fn fail_to_create_values_with_wrong_element_type() {
        let context = create_test_context();

        assert_eq!(
            DenseElementsAttribute::i64_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap_err(),
            Error::ElementExpected {
                r#type: "i64",
                value: "tensor<2xi32>".into()
            }
        );
    }

    #[test]
    fn fail_to_create_values_with_dynamic_shape() {
        let context = create_test_context();

        assert_eq!(
            DenseElementsAttribute::i32_values(
                Type::parse(&context, "tensor<?xi32>").unwrap(),
                &[1]
            )
            .unwrap_err(),
            Error::TypeExpected("statically shaped", "tensor<?xi32>".into())
        );
    }

    #[test]
    fn fail_to_create_strings_with_integer_element_type() {
        let context = create_test_context();

        assert_eq!(
            DenseElementsAttribute::string_values(
                Type::parse(&context, "tensor<1xi8>").unwrap(),
                &[StringRef::new("foo")]
            )
            .unwrap_err(),
            Error::ElementExpected {
                r#type: "string",
                value: "tensor<1xi8>".into()
            }
        );
    }

    #[test]
    fn fail_to_create_with_wrong_attribute() {
        let context = create_test_context();
        let attribute = IntegerAttribute::new(IntegerType::new(&context, 64).into(), 1).into();

        assert_eq!(
            DenseElementsAttribute::new(
                Type::parse(&context, "tensor<1xi8>").unwrap(),
                &[attribute]
            )
            .unwrap_err(),
            Error::AttributeExpected("dense element", attribute.to_string())
        );
    }

    #[test]
    fn fail_to_create_splat_with_wrong_element_type() {
        let context = create_test_context();

        assert_eq!(
            DenseElementsAttribute::f32_splat(i32_tensor_type(&context, 4), 1.0).unwrap_err(),
            Error::ElementExpected {
                r#type: "f32",
                value: "tensor<4xi32>".into()
            }
        );
    }

    #[test]
    fn fail_to_create_from_raw_buffer_with_wrong_size() {
        let context = create_test_context();

        assert!(
            DenseElementsAttribute::raw_buffer(i32_tensor_type(&context, 4), &[0; 3]).is_none()
        );
    }

    #[test]
    fn fail_to_get_element_of_wrong_width() {
        let context = create_test_context();
        let attribute = DenseElementsAttribute::i8_values(
            Type::parse(&context, "tensor<4xi8>").unwrap(),
            &[1, 2, 3, 4],
        )
        .unwrap();

        assert_eq!(
            attribute.i64_element(3),
            Err(Error::ElementExpected {
                r#type: "i64",
                value: attribute.to_string()
            })
        );
    }

    #[test]
    fn fail_to_get_string_element_of_integers() {
        let context = create_test_context();
        let attribute =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap();

        assert_eq!(
            attribute.string_element(0),
            Err(Error::ElementExpected {
                r#type: "string",
                value: attribute.to_string()
            })
        );
    }

    #[test]
    fn fail_to_get_splat_value_of_non_splat() {
        let context = create_test_context();
        let attribute =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap();

        assert_eq!(
            attribute.splat_value().unwrap_err(),
            Error::AttributeExpected("splat dense elements", attribute.to_string())
        );
        assert_eq!(
            attribute.i32_splat_value(),
            Err(Error::AttributeExpected(
                "splat dense elements",
                attribute.to_string()
            ))
        );
    }

    #[test]
    fn fail_to_get_splat_value_of_wrong_type() {
        let context = create_test_context();
        let attribute = DenseElementsAttribute::i32_splat(i32_tensor_type(&context, 2), 1).unwrap();

        assert_eq!(
            attribute.f32_splat_value(),
            Err(Error::ElementExpected {
                r#type: "f32",
                value: attribute.to_string()
            })
        );
    }

    #[test]
    fn fail_to_reshape_with_different_count() {
        let context = create_test_context();
        let attribute =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 6), &[1, 2, 3, 4, 5, 6])
                .unwrap();

        assert_eq!(
            attribute
                .reshape(Type::parse(&context, "tensor<2x2xi32>").unwrap())
                .unwrap_err(),
            Error::CountMismatch {
                name: "dense element",
                expected: 6,
                actual: 4
            }
        );
    }

    #[test]
    fn fail_to_reshape_with_different_element_type() {
        let context = create_test_context();
        let attribute =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]).unwrap();

        assert_eq!(
            attribute.reshape(f32_tensor_type(&context, 2)).unwrap_err(),
            Error::ElementExpected {
                r#type: "the same",
                value: "tensor<2xf32>".into()
            }
        );
    }

    #[test]
    fn fail_to_get_raw_data_of_strings() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let attribute = DenseElementsAttribute::string_values(
            Type::parse(&context, "tensor<1x!foo.string>").unwrap(),
            &[StringRef::new("foo")],
        )
        .unwrap();

        assert_eq!(
            attribute.raw_data(),
            Err(Error::AttributeExpected(
                "numeric dense elements",
                attribute.to_string()
            ))
        );
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseElementsAttribute::i32_values(i32_tensor_type(&context, 3), &[1, 2, 3])
                .unwrap()
                .into();

        assert!(DenseElementsAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseElementsAttribute::try_from(attribute).is_err());
    }
}
