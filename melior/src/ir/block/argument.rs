use super::Value;
use crate::{
    Error,
    ir::{BlockRef, Type, TypeLike, ValueLike},
};
use mlir_sys::{
    MlirValue, mlirBlockArgumentGetArgNumber, mlirBlockArgumentGetOwner, mlirBlockArgumentSetType,
};
use std::fmt::{self, Display, Formatter};

/// A block argument.
#[derive(Clone, Copy, Debug)]
pub struct BlockArgument<'c, 'a> {
    value: Value<'c, 'a>,
}

impl<'c> BlockArgument<'c, '_> {
    /// Returns an argument number.
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.value.to_raw()) as usize }
    }

    /// Returns an owner operation.
    pub fn owner(&self) -> BlockRef<'c, '_> {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.value.to_raw())) }
    }

    /// Sets a type.
    pub fn set_type(&self, r#type: Type) {
        unsafe { mlirBlockArgumentSetType(self.value.to_raw(), r#type.to_raw()) }
    }

    /// Creates a block argument from a raw object.
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

impl<'c> ValueLike<'c> for BlockArgument<'c, '_> {
    fn to_raw(&self) -> MlirValue {
        self.value.to_raw()
    }
}

impl Display for BlockArgument<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Value::from(*self).fmt(formatter)
    }
}

impl<'c, 'a> TryFrom<Value<'c, 'a>> for BlockArgument<'c, 'a> {
    type Error = Error;

    fn try_from(value: Value<'c, 'a>) -> Result<Self, Self::Error> {
        if value.is_block_argument() {
            Ok(Self { value })
        } else {
            Err(Error::BlockArgumentExpected(value.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{Block, Location, block::BlockLike},
    };

    #[test]
    fn argument_number() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(block.argument(0).unwrap().argument_number(), 0);
    }

    #[test]
    fn owner() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert_eq!(&*block.argument(0).unwrap().owner(), &block);
    }

    #[test]
    fn set_type() {
        let context = Context::new();
        let r#type = Type::parse(&context, "index").unwrap();
        let other_type = Type::parse(&context, "f64").unwrap();
        let block = Block::new(&[(r#type, Location::unknown(&context))]);
        let argument = block.argument(0).unwrap();

        argument.set_type(other_type);

        assert_eq!(argument.r#type(), other_type);
    }
}
