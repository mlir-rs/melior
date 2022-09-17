use super::Value;
use crate::ir::{BlockRef, Type};
use mlir_sys::{
    mlirBlockArgumentGetArgNumber, mlirBlockArgumentGetOwner, mlirBlockArgumentSetType,
};
use std::ops::Deref;

/// A block argument.
#[derive(Clone, Copy, Debug)]
pub struct BlockArgument<'a> {
    value: Value<'a>,
}

impl<'a> BlockArgument<'a> {
    pub fn argument_number(&self) -> usize {
        unsafe { mlirBlockArgumentGetArgNumber(self.value.to_raw()) as usize }
    }

    pub fn owner(&self) -> BlockRef {
        unsafe { BlockRef::from_raw(mlirBlockArgumentGetOwner(self.value.to_raw())) }
    }

    pub fn set_type(&self, r#type: Type) {
        unsafe { mlirBlockArgumentSetType(self.value.to_raw(), r#type.to_raw()) }
    }

    pub(crate) unsafe fn from_value(value: Value<'a>) -> Self {
        Self { value }
    }
}

impl<'a> Deref for BlockArgument<'a> {
    type Target = Value<'a>;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{Block, Location},
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

        assert_eq!(block.argument(0).unwrap().owner(), *block);
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
