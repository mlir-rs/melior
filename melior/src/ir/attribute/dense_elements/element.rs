use crate::{
    StringRef,
    ir::{Type, TypeLike, r#type::IntegerType},
};

pub trait DenseElement {
    fn matches(r#type: Type) -> bool;
}

macro_rules! integer_dense_element {
    ($type:ty, $signed:expr) => {
        impl DenseElement for $type {
            fn matches(r#type: Type) -> bool {
                if r#type.is_index() {
                    <$type>::BITS == 64
                } else {
                    IntegerType::try_from(r#type).is_ok_and(|r#type| {
                        r#type.width() == <$type>::BITS
                            && (r#type.is_signless() || r#type.is_signed() == $signed)
                    })
                }
            }
        }
    };
}

macro_rules! float_dense_element {
    ($type:ty, $width:expr) => {
        impl DenseElement for $type {
            fn matches(r#type: Type) -> bool {
                r#type.is_float() && r#type.float_width() == $width
            }
        }
    };
}

integer_dense_element!(i8, true);
integer_dense_element!(i16, true);
integer_dense_element!(i32, true);
integer_dense_element!(i64, true);
integer_dense_element!(u8, false);
integer_dense_element!(u16, false);
integer_dense_element!(u32, false);
integer_dense_element!(u64, false);
float_dense_element!(f32, 32);
float_dense_element!(f64, 64);

impl DenseElement for bool {
    fn matches(r#type: Type) -> bool {
        IntegerType::try_from(r#type).is_ok_and(|r#type| r#type.width() == 1)
    }
}

impl DenseElement for StringRef<'_> {
    fn matches(r#type: Type) -> bool {
        !(r#type.is_integer() || r#type.is_index() || r#type.is_float() || r#type.is_complex())
    }
}
