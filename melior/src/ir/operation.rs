//! Operations and operation builders.

mod builder;
mod operation_like;
mod printing_flags;
mod result;

pub use self::{
    builder::OperationBuilder, printing_flags::OperationPrintingFlags, result::OperationResult,
};
use super::{Attribute, AttributeLike, BlockRef, Identifier, Location, RegionRef, Value};
use crate::{context::Context, utility::print_callback};
use core::{
    fmt,
    mem::{forget, transmute},
};
use mlir_sys::{
    mlirOperationClone, mlirOperationDestroy, mlirOperationEqual, mlirOperationPrint, MlirOperation,
};
pub use operation_like::OperationLike;
use std::{
    ffi::c_void,
    fmt::{Debug, Display, Formatter},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

/// An operation.
pub struct Operation<'c> {
    raw: MlirOperation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Operation<'c> {
    /// Creates an operation from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Creates an optional operation from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    /// Converts an operation into a raw object.
    pub const fn into_raw(self) -> MlirOperation {
        let operation = self.raw;

        forget(self);

        operation
    }
}

impl<'c: 'a, 'a> OperationLike<'c, 'a> for &'a Operation<'c> {
    fn to_raw(self) -> MlirOperation {
        self.raw
    }
}

impl Clone for Operation<'_> {
    fn clone(&self) -> Self {
        unsafe { Self::from_raw(mlirOperationClone(self.raw)) }
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.raw) };
    }
}

impl PartialEq for Operation<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl Eq for Operation<'_> {}

impl Display for Operation<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirOperationPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for Operation<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        writeln!(formatter, "Operation(")?;
        Display::fmt(self, formatter)?;
        write!(formatter, ")")
    }
}

/// A reference to an operation.
#[derive(Clone, Copy)]
pub struct OperationRef<'c, 'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'c>>,
}

impl<'c: 'a, 'a> OperationLike<'c, 'a> for OperationRef<'c, 'a> {
    fn to_raw(self) -> MlirOperation {
        self.raw
    }
}

impl<'c, 'a> OperationRef<'c, 'a> {
    /// Creates an operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    /// Creates an optional operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> Deref for OperationRef<'c, '_> {
    type Target = Operation<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl PartialEq for OperationRef<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl Eq for OperationRef<'_, '_> {}

impl Display for OperationRef<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}

impl Debug for OperationRef<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

/// A mutable reference to an operation.
#[derive(Clone, Copy)]
pub struct OperationRefMut<'c, 'a> {
    raw: MlirOperation,
    _reference: PhantomData<&'a Operation<'c>>,
}

impl<'c: 'a, 'a> OperationLike<'c, 'a> for OperationRefMut<'c, 'a> {
    fn to_raw(self) -> MlirOperation {
        self.raw
    }
}

impl OperationRefMut<'_, '_> {
    /// Creates an operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirOperation) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    /// Creates an optional operation reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirOperation) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }
}

impl<'c> Deref for OperationRefMut<'c, '_> {
    type Target = Operation<'c>;

    fn deref(&self) -> &Self::Target {
        unsafe { transmute(self) }
    }
}

impl DerefMut for OperationRefMut<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { transmute(self) }
    }
}

impl PartialEq for OperationRefMut<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirOperationEqual(self.raw, other.raw) }
    }
}

impl Eq for OperationRefMut<'_, '_> {}

impl Display for OperationRefMut<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self.deref(), formatter)
    }
}

impl Debug for OperationRefMut<'_, '_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Debug::fmt(self.deref(), formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        ir::{attribute::StringAttribute, Block, BlockLike, Location, Region, RegionLike, Type},
        test::create_test_context,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();
    }

    #[test]
    fn name() {
        let context = Context::new();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .name(),
            Identifier::new(&context, "foo")
        );
    }

    #[test]
    fn block() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let block = Block::new(&[]);
        let operation = block.append_operation(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap(),
        );

        assert_eq!(operation.block().as_deref(), Some(&block));
    }

    #[test]
    fn block_none() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
                .block(),
            None
        );
    }

    #[test]
    fn result_error() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
                .result(0)
                .unwrap_err(),
            Error::PositionOutOfBounds {
                name: "operation result",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            }
        );
    }

    #[test]
    fn region_none() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .region(0),
            Err(Error::PositionOutOfBounds {
                name: "region",
                value: "\"foo\"() : () -> ()\n".into(),
                index: 0
            })
        );
    }

    #[test]
    fn operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument: Value = block.argument(0).unwrap().into();

        let operands = vec![argument, argument, argument];
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_operands(&operands)
            .build()
            .unwrap();

        assert_eq!(
            operation.operands().skip(1).collect::<Vec<_>>(),
            vec![argument, argument]
        );
    }

    #[test]
    fn regions() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions([Region::new()])
            .build()
            .unwrap();

        assert_eq!(
            operation.regions().collect::<Vec<_>>(),
            vec![operation.region(0).unwrap()]
        );
    }

    #[test]
    fn location() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let location = Location::new(&context, "test", 1, 1);

        let operation = OperationBuilder::new("foo", location)
            .add_regions([Region::new()])
            .build()
            .unwrap();

        assert_eq!(operation.location(), location);
    }

    #[test]
    fn attribute() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let mut operation = OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                StringAttribute::new(&context, "bar").into(),
            )])
            .build()
            .unwrap();
        assert!(operation.has_attribute("foo"));
        assert_eq!(
            operation.attribute("foo").map(|a| a.to_string()),
            Ok("\"bar\"".into())
        );
        assert!(operation.remove_attribute("foo").is_ok());
        assert!(operation.remove_attribute("foo").is_err());
        operation.set_attribute("foo", StringAttribute::new(&context, "foo").into());
        assert_eq!(
            operation.attribute("foo").map(|a| a.to_string()),
            Ok("\"foo\"".into())
        );
        assert_eq!(
            operation.attributes().next(),
            Some((
                Identifier::new(&context, "foo"),
                StringAttribute::new(&context, "foo").into()
            ))
        )
    }

    #[test]
    fn clone() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);
        let operation = OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();

        let _ = operation.clone();
    }

    #[test]
    fn display() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context),)
                .build()
                .unwrap()
                .to_string(),
            "\"foo\"() : () -> ()\n"
        );
    }

    #[test]
    fn debug() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            format!(
                "{:?}",
                OperationBuilder::new("foo", Location::unknown(&context))
                    .build()
                    .unwrap()
            ),
            "Operation(\n\"foo\"() : () -> ()\n)"
        );
    }

    #[test]
    fn to_string_with_flags() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        assert_eq!(
            OperationBuilder::new("foo", Location::unknown(&context))
                .build()
                .unwrap()
                .to_string_with_flags(
                    OperationPrintingFlags::new()
                        .elide_large_elements_attributes(100)
                        .enable_debug_info(true, true)
                        .print_generic_operation_form()
                        .use_local_scope()
                ),
            Ok("\"foo\"() : () -> () [unknown]".into())
        );
    }

    #[test]
    fn remove_from_parent() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let block = Block::new(&[]);

        let first_operation = block.append_operation(
            OperationBuilder::new("foo", location)
                .add_results(&[Type::index(&context)])
                .build()
                .unwrap(),
        );
        block.append_operation(
            OperationBuilder::new("bar", location)
                .add_operands(&[first_operation.result(0).unwrap().into()])
                .build()
                .unwrap(),
        );
        block.first_operation_mut().unwrap().remove_from_parent();

        assert_eq!(block.first_operation().unwrap().next_in_block(), None);
        assert_eq!(
            block.first_operation().unwrap().to_string(),
            "\"bar\"(<<UNKNOWN SSA VALUE>>) : (index) -> ()"
        );
    }

    #[test]
    fn parent_operation() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let block = Block::new(&[]);

        let operation = block.append_operation(
            OperationBuilder::new("foo", location)
                .add_results(&[Type::index(&context)])
                .add_regions([{
                    let region = Region::new();

                    let block = Block::new(&[]);
                    block.append_operation(OperationBuilder::new("bar", location).build().unwrap());

                    region.append_block(block);
                    region
                }])
                .build()
                .unwrap(),
        );

        assert_eq!(operation.parent_operation(), None);
        assert_eq!(
            &operation
                .region(0)
                .unwrap()
                .first_block()
                .unwrap()
                .first_operation()
                .unwrap()
                .parent_operation()
                .unwrap(),
            &operation
        );
    }
}
