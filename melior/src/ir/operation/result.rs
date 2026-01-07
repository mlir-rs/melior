use crate::{
    Error,
    ir::{OperationRef, Value, ValueLike},
};
use mlir_sys::{MlirValue, mlirOpResultGetOwner, mlirOpResultGetResultNumber};
use std::fmt::{self, Display, Formatter};

/// An operation result.
#[derive(Clone, Copy, Debug)]
pub struct OperationResult<'c, 'a> {
    value: Value<'c, 'a>,
}

impl<'c, 'a> OperationResult<'c, 'a> {
    /// Returns a result number.
    pub fn result_number(&self) -> usize {
        unsafe { mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
    }

    /// Returns an owner operation.
    pub fn owner(&self) -> OperationRef<'c, 'a> {
        unsafe { OperationRef::from_raw(mlirOpResultGetOwner(self.value.to_raw())) }
    }

    /// Creates an operation result from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }
}

impl<'c> ValueLike<'c> for OperationResult<'c, '_> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl Display for OperationResult<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}

impl<'c, 'a> TryFrom<Value<'c, 'a>> for OperationResult<'c, 'a> {
    type Error = Error;

    fn try_from(value: Value<'c, 'a>) -> Result<Self, Self::Error> {
        if value.is_operation_result() {
            Ok(Self { value })
        } else {
            Err(Error::OperationResultExpected(value.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::{
            Location, Type,
            operation::{OperationBuilder, operation_like::OperationLike},
        },
        test::create_test_context,
    };

    #[test]
    fn result_number() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let r#type = Type::parse(&context, "index").unwrap();
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build()
            .unwrap();

        assert_eq!(operation.result(0).unwrap().result_number(), 0);
    }

    #[test]
    fn owner() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let r#type = Type::parse(&context, "index").unwrap();
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build()
            .unwrap();

        let op_ref = operation.result(0).unwrap().owner();
        assert_eq!(*op_ref, operation);
    }
}
