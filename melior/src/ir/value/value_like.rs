use super::Type;
use crate::{
    ContextRef,
    ir::{Location, TypeLike},
};
use mlir_sys::{
    MlirValue, mlirValueDump, mlirValueGetContext, mlirValueGetLocation, mlirValueGetType,
    mlirValueIsABlockArgument, mlirValueIsAOpResult, mlirValueSetType,
};

/// A trait for value-like types.
pub trait ValueLike<'c> {
    /// Converts a value into a raw value.
    fn to_raw(&self) -> MlirValue;

    /// Returns a type.
    fn r#type(&self) -> Type<'c> {
        unsafe { Type::from_raw(mlirValueGetType(self.to_raw())) }
    }

    /// Sets the type of a value.
    fn set_type(&self, r#type: Type<'c>) {
        unsafe { mlirValueSetType(self.to_raw(), r#type.to_raw()) }
    }

    /// Returns the context that owns this value.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirValueGetContext(self.to_raw())) }
    }

    /// Returns the location of this value.
    fn location(&self) -> Location<'c> {
        unsafe { Location::from_raw(mlirValueGetLocation(self.to_raw())) }
    }

    // TODO: expose mlirValuePrintAsOperand once MlirAsmState is available.

    // TODO: expose mlirValueGetFirstUse once MlirOpOperand wrapper type exists.

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
