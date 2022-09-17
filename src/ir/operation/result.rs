use super::Value;
use crate::{
    ir::{OperationRef, ValueLike},
    utility::print_callback,
    Error,
};
use mlir_sys::{mlirOpResultGetOwner, mlirOpResultGetResultNumber, mlirValuePrint, MlirValue};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
};

/// An operation result.
#[derive(Clone, Copy, Debug)]
pub struct ResultValue<'a> {
    value: Value<'a>,
}

impl<'a> ResultValue<'a> {
    pub fn result_number(&self) -> usize {
        unsafe { mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
    }

    pub fn owner(&self) -> OperationRef {
        unsafe { OperationRef::from_raw(mlirOpResultGetOwner(self.value.to_raw())) }
    }
}

impl<'a> ValueLike for ResultValue<'a> {
    unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            value: Value::from_raw(value),
        }
    }

    unsafe fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl<'a> Display for ResultValue<'a> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirValuePrint(
                self.value.to_raw(),
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl<'a> TryFrom<Value<'a>> for ResultValue<'a> {
    type Error = Error;

    fn try_from(value: Value<'a>) -> Result<Self, Self::Error> {
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
        context::Context,
        ir::{operation, Block, Location, Type},
    };

    #[test]
    fn result_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let operation = operation::Builder::new("foo", Location::unknown(&context))
            .add_results(&[r#type])
            .build();

        assert_eq!(operation.result(0).unwrap().result_number(), 0);
    }

    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(block.argument(0).unwrap().owner(), *block);
    }
}
