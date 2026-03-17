use super::{Attribute, AttributeLike};
use crate::{
    Error, StringRef,
    ir::{Type, TypeLike},
};
use mlir_sys::{
    MlirAttribute, MlirStringRef, mlirDenseElementsAttrBoolGet, mlirDenseElementsAttrBoolSplatGet,
    mlirDenseElementsAttrDoubleGet, mlirDenseElementsAttrDoubleSplatGet,
    mlirDenseElementsAttrFloatGet, mlirDenseElementsAttrFloatSplatGet, mlirDenseElementsAttrGet,
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
    mlirDenseElementsAttrUInt64SplatGet, mlirElementsAttrGetNumElements,
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

    /// Creates a dense elements attribute from bool values.
    ///
    /// The C API takes `int*`, so values are passed as `&[i32]` where non-zero
    /// means true.
    pub fn bool_values(r#type: Type<'c>, values: &[i32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrBoolGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from i8 values.
    pub fn i8_values(r#type: Type<'c>, values: &[i8]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrInt8Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from i16 values.
    pub fn i16_values(r#type: Type<'c>, values: &[i16]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrInt16Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from i32 values.
    pub fn i32_values(r#type: Type<'c>, values: &[i32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrInt32Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from i64 values.
    pub fn i64_values(r#type: Type<'c>, values: &[i64]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrInt64Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from u8 values.
    pub fn u8_values(r#type: Type<'c>, values: &[u8]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrUInt8Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from u16 values.
    pub fn u16_values(r#type: Type<'c>, values: &[u16]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrUInt16Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from u32 values.
    pub fn u32_values(r#type: Type<'c>, values: &[u32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrUInt32Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from u64 values.
    pub fn u64_values(r#type: Type<'c>, values: &[u64]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrUInt64Get(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from f32 values.
    pub fn f32_values(r#type: Type<'c>, values: &[f32]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrFloatGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from f64 values.
    pub fn f64_values(r#type: Type<'c>, values: &[f64]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrDoubleGet(
                r#type.to_raw(),
                values.len() as isize,
                values.as_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from string values.
    pub fn string_values(r#type: Type<'c>, values: &[StringRef]) -> Self {
        let mut raws: Vec<MlirStringRef> = values.iter().map(|s| s.to_raw()).collect();
        unsafe {
            Self::from_raw(mlirDenseElementsAttrStringGet(
                r#type.to_raw(),
                raws.len() as isize,
                raws.as_mut_ptr(),
            ))
        }
    }

    /// Creates a dense elements attribute from a raw buffer of packed element
    /// data.
    pub fn raw_buffer(r#type: Type<'c>, buffer: &[u8]) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrRawBufferGet(
                r#type.to_raw(),
                buffer.len(),
                buffer.as_ptr() as *const _,
            ))
        }
    }

    // -------------------------------------------------------------------------
    // Splat constructors
    // -------------------------------------------------------------------------

    /// Creates a splat dense elements attribute with the given element
    /// attribute.
    pub fn splat(r#type: Type<'c>, element: Attribute<'c>) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrSplatGet(
                r#type.to_raw(),
                element.to_raw(),
            ))
        }
    }

    /// Creates a splat dense elements attribute with a bool value.
    pub fn bool_splat(r#type: Type<'c>, value: bool) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrBoolSplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with an i8 value.
    pub fn i8_splat(r#type: Type<'c>, value: i8) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrInt8SplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with an i32 value.
    pub fn i32_splat(r#type: Type<'c>, value: i32) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrInt32SplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with an i64 value.
    pub fn i64_splat(r#type: Type<'c>, value: i64) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrInt64SplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with a u32 value.
    pub fn u32_splat(r#type: Type<'c>, value: u32) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrUInt32SplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with a u64 value.
    pub fn u64_splat(r#type: Type<'c>, value: u64) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrUInt64SplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with an f32 value.
    pub fn f32_splat(r#type: Type<'c>, value: f32) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrFloatSplatGet(r#type.to_raw(), value)) }
    }

    /// Creates a splat dense elements attribute with an f64 value.
    pub fn f64_splat(r#type: Type<'c>, value: f64) -> Self {
        unsafe { Self::from_raw(mlirDenseElementsAttrDoubleSplatGet(r#type.to_raw(), value)) }
    }

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
    pub fn reshape(&self, r#type: Type<'c>) -> Self {
        unsafe {
            Self::from_raw(mlirDenseElementsAttrReshapeGet(
                self.attribute.to_raw(),
                r#type.to_raw(),
            ))
        }
    }

    /// Returns a raw pointer to the packed element data.
    pub fn raw_data(&self) -> *const std::ffi::c_void {
        unsafe { mlirDenseElementsAttrGetRawData(self.attribute.to_raw()) }
    }

    // -------------------------------------------------------------------------
    // Element accessors
    // -------------------------------------------------------------------------

    /// Returns a bool element at the given index.
    pub fn bool_element(&self, index: usize) -> Result<bool, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetBoolValue(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an i8 element at the given index.
    pub fn i8_element(&self, index: usize) -> Result<i8, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetInt8Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an i16 element at the given index.
    pub fn i16_element(&self, index: usize) -> Result<i16, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetInt16Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
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

    /// Returns a u8 element at the given index.
    pub fn u8_element(&self, index: usize) -> Result<u8, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetUInt8Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns a u16 element at the given index.
    pub fn u16_element(&self, index: usize) -> Result<u16, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetUInt16Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns a u32 element at the given index.
    pub fn u32_element(&self, index: usize) -> Result<u32, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetUInt32Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns a u64 element at the given index.
    pub fn u64_element(&self, index: usize) -> Result<u64, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetUInt64Value(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an f32 element at the given index.
    pub fn f32_element(&self, index: usize) -> Result<f32, Error> {
        if !self.is_dense_fp_elements() {
            Err(Error::ElementExpected {
                r#type: "floating point",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetFloatValue(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an f64 element at the given index.
    pub fn f64_element(&self, index: usize) -> Result<f64, Error> {
        if !self.is_dense_fp_elements() {
            Err(Error::ElementExpected {
                r#type: "floating point",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetDoubleValue(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns an index element at the given index.
    pub fn index_element(&self, index: usize) -> Result<u64, Error> {
        if !self.is_dense_int_elements() {
            Err(Error::ElementExpected {
                r#type: "integer",
                value: self.to_string(),
            })
        } else if index < self.len() {
            Ok(unsafe {
                mlirDenseElementsAttrGetIndexValue(self.attribute.to_raw(), index as isize)
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dense element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns a string element at the given index.
    pub fn string_element(&self, index: usize) -> Result<&str, Error> {
        if index < self.len() {
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
    pub fn splat_value(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirDenseElementsAttrGetSplatValue(self.attribute.to_raw())) }
    }

    /// Returns the bool splat value. Non-zero means true.
    pub fn bool_splat_value(&self) -> bool {
        unsafe { mlirDenseElementsAttrGetBoolSplatValue(self.attribute.to_raw()) != 0 }
    }

    /// Returns the i8 splat value.
    pub fn i8_splat_value(&self) -> i8 {
        unsafe { mlirDenseElementsAttrGetInt8SplatValue(self.attribute.to_raw()) }
    }

    /// Returns the i32 splat value.
    pub fn i32_splat_value(&self) -> i32 {
        unsafe { mlirDenseElementsAttrGetInt32SplatValue(self.attribute.to_raw()) }
    }

    /// Returns the i64 splat value.
    pub fn i64_splat_value(&self) -> i64 {
        unsafe { mlirDenseElementsAttrGetInt64SplatValue(self.attribute.to_raw()) }
    }

    /// Returns the u32 splat value.
    pub fn u32_splat_value(&self) -> u32 {
        unsafe { mlirDenseElementsAttrGetUInt32SplatValue(self.attribute.to_raw()) }
    }

    /// Returns the u64 splat value.
    pub fn u64_splat_value(&self) -> u64 {
        unsafe { mlirDenseElementsAttrGetUInt64SplatValue(self.attribute.to_raw()) }
    }

    /// Returns the f32 splat value.
    pub fn f32_splat_value(&self) -> f32 {
        unsafe { mlirDenseElementsAttrGetFloatSplatValue(self.attribute.to_raw()) }
    }

    /// Returns the f64 splat value.
    pub fn f64_splat_value(&self) -> f64 {
        unsafe { mlirDenseElementsAttrGetDoubleSplatValue(self.attribute.to_raw()) }
    }

    /// Returns the string splat value.
    pub fn string_splat_value(&self) -> &str {
        let raw = unsafe { mlirDenseElementsAttrGetStringSplatValue(self.attribute.to_raw()) };
        unsafe { StringRef::from_raw(raw) }
            .as_str()
            .expect("splat string value is valid UTF-8")
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
        let attr = DenseElementsAttribute::i32_values(i32_tensor_type(&context, 3), &[10, 20, 30]);
        assert_eq!(attr.i32_element(0), Ok(10));
        assert_eq!(attr.i32_element(1), Ok(20));
        assert_eq!(attr.i32_element(2), Ok(30));
    }

    #[test]
    fn i64_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::i64_values(i64_tensor_type(&context, 2), &[100i64, 200i64]);
        assert_eq!(attr.i64_element(0), Ok(100));
        assert_eq!(attr.i64_element(1), Ok(200));
    }

    #[test]
    fn i8_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::i8_values(ty, &[1i8, -1i8]);
        assert_eq!(attr.i8_element(0), Ok(1));
        assert_eq!(attr.i8_element(1), Ok(-1));
    }

    #[test]
    fn i16_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 16).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::i16_values(ty, &[300i16, -300i16]);
        assert_eq!(attr.i16_element(0), Ok(300));
        assert_eq!(attr.i16_element(1), Ok(-300));
    }

    #[test]
    fn u8_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        let attr = DenseElementsAttribute::u8_values(ty, &[0u8, 128u8, 255u8]);
        assert_eq!(attr.u8_element(0), Ok(0));
        assert_eq!(attr.u8_element(1), Ok(128));
        assert_eq!(attr.u8_element(2), Ok(255));
    }

    #[test]
    fn u16_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 16).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u16_values(ty, &[1000u16, 2000u16]);
        assert_eq!(attr.u16_element(0), Ok(1000));
        assert_eq!(attr.u16_element(1), Ok(2000));
    }

    #[test]
    fn u32_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u32_values(ty, &[7u32, 42u32]);
        assert_eq!(attr.u32_element(0), Ok(7));
        assert_eq!(attr.u32_element(1), Ok(42));
    }

    #[test]
    fn u64_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 64).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u64_values(ty, &[999u64, 1000u64]);
        assert_eq!(attr.u64_element(0), Ok(999));
        assert_eq!(attr.u64_element(1), Ok(1000));
    }

    #[test]
    fn f32_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f32_values(f32_tensor_type(&context, 2), &[1.5f32, -2.5f32]);
        assert_eq!(attr.f32_element(0), Ok(1.5f32));
        assert_eq!(attr.f32_element(1), Ok(-2.5f32));
    }

    #[test]
    fn f64_values_and_elements() {
        let context = create_test_context();
        let attr =
            DenseElementsAttribute::f64_values(f64_tensor_type(&context, 2), &[3.14f64, -2.72f64]);
        assert_eq!(attr.f64_element(0), Ok(3.14f64));
        assert_eq!(attr.f64_element(1), Ok(-2.72f64));
    }

    #[test]
    fn bool_values_and_elements() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 1).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        // Non-zero = true, 0 = false
        let attr = DenseElementsAttribute::bool_values(ty, &[1, 0, 1]);
        assert_eq!(attr.bool_element(0), Ok(true));
        assert_eq!(attr.bool_element(1), Ok(false));
        assert_eq!(attr.bool_element(2), Ok(true));
    }

    #[test]
    fn string_values_and_elements() {
        let context = create_test_context();
        // MLIR dense string attrs work with any shaped type.
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let strings = [StringRef::new("hello"), StringRef::new("world")];
        let attr = DenseElementsAttribute::string_values(ty, &strings);
        assert_eq!(attr.string_element(0), Ok("hello"));
        assert_eq!(attr.string_element(1), Ok("world"));
    }

    #[test]
    fn splat_i32() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i32_splat(i32_tensor_type(&context, 4), 77);
        assert!(attr.is_splat());
        assert_eq!(attr.i32_splat_value(), 77);
        assert_eq!(attr.i32_element(0), Ok(77));
        assert_eq!(attr.i32_element(3), Ok(77));
    }

    #[test]
    fn splat_i64() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i64_splat(i64_tensor_type(&context, 3), 999i64);
        assert!(attr.is_splat());
        assert_eq!(attr.i64_splat_value(), 999);
    }

    #[test]
    fn splat_u32() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let ty = RankedTensorType::new(&[5], element, None).into();
        let attr = DenseElementsAttribute::u32_splat(ty, 42u32);
        assert!(attr.is_splat());
        assert_eq!(attr.u32_splat_value(), 42);
    }

    #[test]
    fn splat_u64() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 64).into();
        let ty = RankedTensorType::new(&[2], element, None).into();
        let attr = DenseElementsAttribute::u64_splat(ty, 12345u64);
        assert!(attr.is_splat());
        assert_eq!(attr.u64_splat_value(), 12345);
    }

    #[test]
    fn splat_f32() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::f32_splat(f32_tensor_type(&context, 4), 3.14f32);
        assert!(attr.is_splat());
        assert_eq!(attr.f32_splat_value(), 3.14f32);
    }

    #[test]
    fn splat_f64() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::f64_splat(f64_tensor_type(&context, 2), 2.72f64);
        assert!(attr.is_splat());
        assert_eq!(attr.f64_splat_value(), 2.72f64);
    }

    #[test]
    fn splat_bool() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 1).into();
        let ty = RankedTensorType::new(&[3], element, None).into();
        let attr = DenseElementsAttribute::bool_splat(ty, true);
        assert!(attr.is_splat());
        assert!(attr.bool_splat_value());
    }

    #[test]
    fn splat_i8() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 8).into();
        let ty = RankedTensorType::new(&[4], element, None).into();
        let attr = DenseElementsAttribute::i8_splat(ty, -5i8);
        assert!(attr.is_splat());
        assert_eq!(attr.i8_splat_value(), -5);
    }

    #[test]
    fn splat_attribute() {
        let context = create_test_context();
        let integer_type = IntegerType::new(&context, 32);
        let ty = i32_tensor_type(&context, 3);
        let elem = IntegerAttribute::new(integer_type.into(), 55).into();
        let attr = DenseElementsAttribute::splat(ty, elem);
        assert!(attr.is_splat());
        assert_eq!(attr.i32_element(0), Ok(55));
    }

    #[test]
    fn raw_buffer_round_trip() {
        let context = create_test_context();
        // Build a tensor of 4 x i32 from a raw byte buffer
        let values: [i32; 4] = [1, 2, 3, 4];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
        let ty = i32_tensor_type(&context, 4);
        let attr = DenseElementsAttribute::raw_buffer(ty, bytes);
        assert_eq!(attr.len(), 4);
        assert!(!attr.raw_data().is_null());
    }

    #[test]
    fn reshape() {
        let context = create_test_context();
        let element = IntegerType::new(&context, 32).into();
        let original_ty = RankedTensorType::new(&[6], element, None).into();
        let attr = DenseElementsAttribute::i32_values(original_ty, &[1, 2, 3, 4, 5, 6]);
        let new_ty = RankedTensorType::new(&[2, 3], element, None).into();
        let reshaped = attr.reshape(new_ty);
        assert_eq!(reshaped.len(), 6);
        assert_eq!(reshaped.i32_element(0), Ok(1));
        assert_eq!(reshaped.i32_element(5), Ok(6));
    }

    #[test]
    fn out_of_bounds_error() {
        let context = create_test_context();
        let attr = DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]);
        assert!(matches!(
            attr.i32_element(5),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn wrong_type_error() {
        let context = create_test_context();
        // Create a float tensor and try accessing as int
        let attr = DenseElementsAttribute::f32_values(f32_tensor_type(&context, 2), &[1.0, 2.0]);
        assert!(matches!(
            attr.i32_element(0),
            Err(Error::ElementExpected { .. })
        ));
        // Create an int tensor and try accessing as float
        let attr2 = DenseElementsAttribute::i32_values(i32_tensor_type(&context, 2), &[1, 2]);
        assert!(matches!(
            attr2.f32_element(0),
            Err(Error::ElementExpected { .. })
        ));
    }
}
