use super::Type;
use mlir_sys::{
    MlirValue, mlirValueDump, mlirValueGetType, mlirValueIsABlockArgument, mlirValueIsAOpResult,
};

/// A trait for value-like types.
pub trait ValueLike<'c> {
    /// Converts a value into a raw value.
    fn to_raw(&self) -> MlirValue;

    /// Returns a type.
    fn r#type(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirValueGetType(self.to_raw())) }
    }

    /// Returns `true` if a value is a block argument.
    fn is_block_argument(&self) -> bool {
        unsafe { mlirValueIsABlockArgument(self.to_raw()) }
    }

    /// Returns `true` if a value is an operation result.
    fn is_operation_result(&self) -> bool {
        unsafe { mlirValueIsAOpResult(self.to_raw()) }
    }

    /// Dumps a value.
    fn dump(&self) {
        unsafe { mlirValueDump(self.to_raw()) }
    }
}
