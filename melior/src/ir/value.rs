mod value_like;

pub use self::value_like::ValueLike;
use super::{Type, block::BlockArgument, operation::OperationResult};
use crate::{Context, utility::print_callback};
use mlir_sys::{MlirValue, mlirValueEqual, mlirValuePrint};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// A value.
// Values are always non-owning references to their parents, such as operations
// and blocks. See the `Value` class in the MLIR C++ API.
#[derive(Clone, Copy)]
pub struct Value<'c, 'a> {
    raw: MlirValue,
    _context: PhantomData<&'c Context>,
    _parent: PhantomData<&'a ()>,
}

impl Value<'_, '_> {
    /// Creates a value from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(value: MlirValue) -> Self {
        Self {
            raw: value,
            _context: Default::default(),
            _parent: Default::default(),
        }
    }
}

impl<'c> ValueLike<'c> for Value<'c, '_> {
    fn to_raw(&self) -> MlirValue {
        self.raw
    }
}

impl PartialEq for Value<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirValueEqual(self.raw, other.raw) }
    }
}

impl Eq for Value<'_, '_> {}

impl Display for Value<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirValuePrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Value<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Value(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

from_borrowed_subtypes!(Value, BlockArgument, OperationResult);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context,
        ir::{
            Attribute, Block, Identifier, Location,
            block::BlockLike,
            operation::{OperationBuilder, OperationLike},
        },
        test::create_test_context,
    };

    #[test]
    fn r#type() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        assert_eq!(operation.result(0).unwrap().r#type(), index_type);
    }

    #[test]
    fn is_operation_result() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let r#type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[r#type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        assert!(operation.result(0).unwrap().is_operation_result());
    }

    #[test]
    fn is_block_argument() {
        let context = create_test_context();
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, Location::unknown(&context))]);

        assert!(block.argument(0).unwrap().is_block_argument());
    }

    #[test]
    fn dump() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let value = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        value.result(0).unwrap().dump();
    }

    #[test]
    fn equal() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();
        let result = Value::from(operation.result(0).unwrap());

        assert_eq!(result, result);
    }

    #[test]
    fn not_equal() {
        let context = create_test_context();
        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = || {
            OperationBuilder::new("arith.constant", location)
                .add_results(&[index_type])
                .add_attributes(&[(
                    Identifier::new(&context, "value"),
                    Attribute::parse(&context, "0 : index").unwrap(),
                )])
                .build()
                .unwrap()
        };

        assert_ne!(
            Value::from(operation().result(0).unwrap()),
            operation().result(0).unwrap().into()
        );
    }

    #[test]
    fn display_with_unregistered_dialect() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%0 = \"arith.constant\"() {value = 0 : index} : () -> index\n"
        );
    }

    #[test]
    fn display_with_registered_dialect() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        assert_eq!(
            operation.result(0).unwrap().to_string(),
            "%c0 = arith.constant 0 : index\n"
        );
    }

    #[test]
    fn debug() {
        let context = create_test_context();

        let location = Location::unknown(&context);
        let index_type = Type::index(&context);

        let operation = OperationBuilder::new("arith.constant", location)
            .add_results(&[index_type])
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                Attribute::parse(&context, "0 : index").unwrap(),
            )])
            .build()
            .unwrap();

        assert_eq!(
            format!("{:?}", Value::from(operation.result(0).unwrap())),
            "Value(\n%c0 = arith.constant 0 : index\n)"
        );
    }
}
