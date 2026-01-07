use super::TypeId;
use crate::{context::ContextRef, dialect::Dialect};
use mlir_sys::{MlirType, mlirTypeDump, mlirTypeGetContext, mlirTypeGetDialect, mlirTypeGetTypeID};

/// A trait for type-like types.
pub trait TypeLike<'c> {
    /// Converts a type into a raw object.
    fn to_raw(&self) -> MlirType;

    /// Returns a context.
    fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirTypeGetContext(self.to_raw())) }
    }

    /// Returns an ID.
    fn id(&self) -> TypeId<'c> {
        unsafe { TypeId::from_raw(mlirTypeGetTypeID(self.to_raw())) }
    }

    /// Returns a dialect.
    fn dialect(&self) -> Dialect<'c> {
        unsafe { Dialect::from_raw(mlirTypeGetDialect(self.to_raw())) }
    }

    /// Dumps a type.
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_raw()) }
    }

    melior_macro::type_check_functions!(
        // spell-checker: disable
        mlirTypeIsAAnyQuantizedType,
        mlirTypeIsABF16,
        mlirTypeIsACalibratedQuantizedType,
        mlirTypeIsAComplex,
        mlirTypeIsAF16,
        mlirTypeIsAF32,
        mlirTypeIsAF64,
        mlirTypeIsAFloat,
        mlirTypeIsAFloat8E4M3,
        mlirTypeIsAFloat8E4M3B11FNUZ,
        mlirTypeIsAFloat8E4M3FN,
        mlirTypeIsAFloat8E4M3FNUZ,
        mlirTypeIsAFloat8E5M2,
        mlirTypeIsAFloat8E5M2FNUZ,
        mlirTypeIsAFunction,
        mlirTypeIsAGPUAsyncTokenType,
        mlirTypeIsAIndex,
        mlirTypeIsAInteger,
        mlirTypeIsALLVMPointerType,
        mlirTypeIsALLVMStructType,
        mlirTypeIsAMemRef,
        mlirTypeIsANone,
        mlirTypeIsANVGPUTensorMapDescriptorType,
        mlirTypeIsAOpaque,
        mlirTypeIsAPDLAttributeType,
        mlirTypeIsAPDLOperationType,
        mlirTypeIsAPDLRangeType,
        mlirTypeIsAPDLType,
        mlirTypeIsAPDLTypeType,
        mlirTypeIsAPDLValueType,
        mlirTypeIsAQuantizedType,
        mlirTypeIsARankedTensor,
        mlirTypeIsAShaped,
        mlirTypeIsATensor,
        mlirTypeIsATF32,
        mlirTypeIsATransformAnyOpType,
        mlirTypeIsATransformAnyParamType,
        mlirTypeIsATransformAnyValueType,
        mlirTypeIsATransformOperationType,
        mlirTypeIsATransformParamType,
        mlirTypeIsATuple,
        mlirTypeIsAUniformQuantizedPerAxisType,
        mlirTypeIsAUniformQuantizedType,
        mlirTypeIsAUnrankedMemRef,
        mlirTypeIsAUnrankedTensor,
        mlirTypeIsAVector,
        // spell-checker: enable
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        ir::{
            Type,
            r#type::{FunctionType, IntegerType},
        },
    };

    #[test]
    fn context() {
        Type::parse(&Context::new(), "i8").unwrap().context();
    }

    #[test]
    fn id() {
        let context = Context::new();

        assert_eq!(Type::index(&context).id(), Type::index(&context).id());
    }

    #[test]
    fn dialect() {
        let context = Context::new();

        assert_eq!(
            Type::index(&context).dialect().namespace().unwrap(),
            "builtin"
        );
    }

    #[test]
    fn is_integer() {
        let context = Context::new();

        assert!(IntegerType::new(&context, 64).is_integer());
    }

    #[test]
    fn is_index() {
        let context = Context::new();

        assert!(Type::index(&context).is_index());
    }

    #[test]
    fn is_bfloat16() {
        let context = Context::new();

        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_function() {
        let context = Context::new();

        assert!(FunctionType::new(&context, &[], &[]).is_function());
    }

    #[test]
    fn is_vector() {
        let context = Context::new();

        assert!(Type::vector(&[42], Type::index(&context)).is_vector());
    }

    #[test]
    fn dump() {
        Type::index(&Context::new()).dump();
    }
}
