use super::TypeLike;
use crate::{ir::Type, Context, Error};
use mlir_sys::{
    mlirIntegerTypeGet, mlirIntegerTypeGetWidth, mlirIntegerTypeIsSigned,
    mlirIntegerTypeIsSignless, mlirIntegerTypeIsUnsigned, mlirIntegerTypeSignedGet,
    mlirIntegerTypeUnsignedGet, MlirType,
};

/// A integer type.
#[derive(Clone, Copy, Debug, Hash)]
pub struct IntegerType<'c> {
    r#type: Type<'c>,
}

impl<'c> IntegerType<'c> {
    /// Creates an integer type.
    pub fn new(context: &'c Context, bits: u32) -> Self {
        Self {
            r#type: unsafe { Type::from_raw(mlirIntegerTypeGet(context.to_raw(), bits)) },
        }
    }

    /// Creates a signed integer type.
    pub fn signed(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeSignedGet(context.to_raw(), bits)) }
    }

    /// Creates an unsigned integer type.
    pub fn unsigned(context: &'c Context, bits: u32) -> Self {
        unsafe { Self::from_raw(mlirIntegerTypeUnsignedGet(context.to_raw(), bits)) }
    }

    /// Returns a bit width.
    pub fn width(&self) -> u32 {
        unsafe { mlirIntegerTypeGetWidth(self.to_raw()) }
    }

    /// Checks if an integer type is signed.
    pub fn is_signed(&self) -> bool {
        unsafe { mlirIntegerTypeIsSigned(self.to_raw()) }
    }

    /// Checks if an integer type is signless.
    pub fn is_signless(&self) -> bool {
        unsafe { mlirIntegerTypeIsSignless(self.to_raw()) }
    }

    /// Checks if an integer type is unsigned.
    pub fn is_unsigned(&self) -> bool {
        unsafe { mlirIntegerTypeIsUnsigned(self.to_raw()) }
    }
}

type_traits!(IntegerType, is_integer, "integer");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        assert!(IntegerType::new(&Context::new(), 64).is_integer());
    }

    #[test]
    fn signed() {
        assert!(IntegerType::signed(&Context::new(), 64).is_integer());
    }

    #[test]
    fn unsigned() {
        assert!(IntegerType::unsigned(&Context::new(), 64).is_integer());
    }

    #[test]
    fn signed_integer() {
        let context = Context::new();

        assert_eq!(
            Type::from(IntegerType::signed(&context, 42)),
            Type::parse(&context, "si42").unwrap()
        );
    }

    #[test]
    fn unsigned_integer() {
        let context = Context::new();

        assert_eq!(
            Type::from(IntegerType::unsigned(&context, 42)),
            Type::parse(&context, "ui42").unwrap()
        );
    }

    #[test]
    fn get_width() {
        let context = Context::new();

        assert_eq!(IntegerType::new(&context, 64).width(), 64);
    }

    #[test]
    fn check_sign() {
        let context = Context::new();
        let signless = IntegerType::new(&context, 42);
        let signed = IntegerType::signed(&context, 42);
        let unsigned = IntegerType::unsigned(&context, 42);

        assert!(signless.is_signless());
        assert!(!signed.is_signless());
        assert!(!unsigned.is_signless());

        assert!(!signless.is_signed());
        assert!(signed.is_signed());
        assert!(!unsigned.is_signed());

        assert!(!signless.is_unsigned());
        assert!(!signed.is_unsigned());
        assert!(unsigned.is_unsigned());
    }
}
