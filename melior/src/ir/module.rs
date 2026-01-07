use super::{BlockRef, Location, Operation, OperationRef, operation::OperationRefMut};
use crate::{
    context::{Context, ContextRef},
    string_ref::StringRef,
};
use mlir_sys::{
    MlirModule, mlirModuleCreateEmpty, mlirModuleCreateParse, mlirModuleDestroy,
    mlirModuleFromOperation, mlirModuleGetBody, mlirModuleGetContext, mlirModuleGetOperation,
};
use std::{ffi::CString, marker::PhantomData};

/// A module.
#[derive(Debug)]
pub struct Module<'c> {
    raw: MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    /// Creates a module.
    pub fn new(location: Location) -> Self {
        unsafe { Self::from_raw(mlirModuleCreateEmpty(location.to_raw())) }
    }

    /// Parses a module.
    pub fn parse(context: &Context, source: &str) -> Option<Self> {
        // TODO Use a string not null-terminated.
        // Somehow, we still need a null-terminated string for a source.
        let source = CString::new(source).unwrap();
        let source = StringRef::from_c_str(&source);

        unsafe { Self::from_option_raw(mlirModuleCreateParse(context.to_raw(), source.to_raw())) }
    }

    /// Converts a module into an operation.
    pub fn as_operation(&self) -> OperationRef<'c, '_> {
        unsafe { OperationRef::from_raw(mlirModuleGetOperation(self.raw)) }
    }

    /// Converts a module into a mutable operation.
    pub fn as_operation_mut(&mut self) -> OperationRefMut<'c, '_> {
        unsafe { OperationRefMut::from_raw(mlirModuleGetOperation(self.raw)) }
    }

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirModuleGetContext(self.raw)) }
    }

    /// Returns a block of a module body.
    pub fn body(&self) -> BlockRef<'c, '_> {
        unsafe { BlockRef::from_raw(mlirModuleGetBody(self.raw)) }
    }

    /// Converts an operation into a module.
    pub fn from_operation(operation: Operation) -> Option<Self> {
        unsafe { Self::from_option_raw(mlirModuleFromOperation(operation.into_raw())) }
    }

    /// Creates a module from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirModule) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Creates an optional module from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_option_raw(raw: MlirModule) -> Option<Self> {
        if raw.ptr.is_null() {
            None
        } else {
            Some(Self::from_raw(raw))
        }
    }

    /// Converts a module into a raw object.
    pub const fn to_raw(&self) -> MlirModule {
        self.raw
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            Block, Region, RegionLike,
            attribute::StringAttribute,
            operation::{OperationBuilder, OperationLike, OperationMutLike},
        },
        test::create_test_context,
    };

    #[test]
    fn new() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42));
    }

    #[test]
    fn context() {
        Module::new(Location::new(&Context::new(), "foo", 42, 42)).context();
    }

    #[test]
    fn parse() {
        assert!(Module::parse(&Context::new(), "module{}").is_some());
    }

    #[test]
    fn parse_none() {
        assert!(Module::parse(&Context::new(), "module{").is_none());
    }

    #[test]
    fn from_operation() {
        let context = create_test_context();

        let region = Region::new();
        region.append_block(Block::new(&[]));

        let module = Module::from_operation(
            OperationBuilder::new("builtin.module", Location::unknown(&context))
                .add_regions([region])
                .build()
                .unwrap(),
        )
        .unwrap();

        assert!(module.as_operation().verify());
        assert_eq!(module.as_operation().to_string(), "module {\n}\n")
    }

    #[test]
    fn from_operation_fail() {
        let context = create_test_context();

        assert!(
            Module::from_operation(
                OperationBuilder::new("func.func", Location::unknown(&context),)
                    .build()
                    .unwrap()
            )
            .is_none()
        );
    }

    #[test]
    fn set_attribute() {
        let context = create_test_context();

        let mut module = Module::new(Location::unknown(&context));

        module
            .as_operation_mut()
            .set_attribute("sym_name", StringAttribute::new(&context, "foo").into());

        assert!(module.as_operation().verify());
    }
}
