//! External passes

use super::Pass;
use crate::{
    dialect::DialectHandle,
    ir::{r#type::TypeId, OperationRef},
    ContextRef, StringRef,
};
use mlir_sys::{
    mlirCreateExternalPass, mlirExternalPassSignalFailure, MlirContext, MlirExternalPass,
    MlirExternalPassCallbacks, MlirLogicalResult, MlirOperation,
};
use std::{ffi::c_void, marker::PhantomData, mem::transmute, ptr::drop_in_place};

#[derive(Clone, Copy, Debug)]
pub struct ExternalPass<'a> {
    raw: MlirExternalPass,
    _reference: PhantomData<&'a MlirExternalPass>,
}

impl ExternalPass<'_> {
    /// Signals that the pass has failed.
    pub fn signal_failure(self) {
        unsafe { mlirExternalPassSignalFailure(self.raw) }
    }

    /// Converts an external pass to a raw object.
    pub const fn to_raw(self) -> MlirExternalPass {
        self.raw
    }

    /// Creates an external pass handle from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirExternalPass) -> Self {
        Self {
            raw,
            _reference: PhantomData,
        }
    }
}

unsafe extern "C" fn callback_construct<'a, T: RunExternalPass<'a>>(pass: *mut T) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .construct();
}

unsafe extern "C" fn callback_destruct<'a, T: RunExternalPass<'a>>(pass: *mut T) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .destruct();
    drop_in_place(pass);
}

unsafe extern "C" fn callback_initialize<'a, T: RunExternalPass<'a>>(
    context: MlirContext,
    pass: *mut T,
) -> MlirLogicalResult {
    pass.as_mut()
        .expect("pass should be valid when called")
        .initialize(ContextRef::from_raw(context));

    MlirLogicalResult { value: 1 }
}

unsafe extern "C" fn callback_run<'a, T: RunExternalPass<'a>>(
    operation: MlirOperation,
    mlir_pass: MlirExternalPass,
    pass: *mut T,
) {
    pass.as_mut()
        .expect("pass should be valid when called")
        .run(
            OperationRef::from_raw(operation),
            ExternalPass::from_raw(mlir_pass),
        )
}

unsafe extern "C" fn callback_clone<'a, T: RunExternalPass<'a>>(pass: *mut T) -> *mut T {
    Box::<T>::into_raw(Box::new(
        pass.as_mut()
            .expect("pass should be valid when called")
            .clone(),
    ))
}

/// A trait for MLIR passes written in Rust.
///
/// This trait is implemented for any type that implements `FnMut`,
/// but can be implemented for any struct that implements `Clone`.
///
/// # Examples
///
/// The following example pass dumps operations.
///
/// ```
/// use melior::{
///     ir::{operation::OperationLike, OperationRef},
///     pass::{ExternalPass, RunExternalPass},
///     ContextRef,
/// };
///
/// #[derive(Clone, Debug)]
/// struct ExamplePass;
///
/// impl<'c> RunExternalPass<'c> for ExamplePass {
///     fn construct(&mut self) {
///         println!("Constructed pass!");
///     }
///
///     fn initialize(&mut self, context: ContextRef<'c>) {
///         println!("Initialize called!");
///     }
///
///     fn run(&mut self, operation: OperationRef<'c, '_>, _pass: ExternalPass<'_>) {
///         operation.dump();
///     }
/// }
/// ```
pub trait RunExternalPass<'c>: Sized + Clone {
    fn construct(&mut self) {}
    fn destruct(&mut self) {}
    fn initialize(&mut self, context: ContextRef<'c>);
    fn run(&mut self, operation: OperationRef<'c, '_>, pass: ExternalPass<'_>);
}

impl<'c, F: FnMut(OperationRef<'c, '_>, ExternalPass<'_>) + Clone> RunExternalPass<'c> for F {
    fn initialize(&mut self, _context: ContextRef<'c>) {}

    fn run(&mut self, operation: OperationRef<'c, '_>, pass: ExternalPass<'_>) {
        self(operation, pass)
    }
}

/// Creates a `Pass` object from an external pass
///
/// # Examples
///
/// ```
/// use melior::{
///     ir::{operation::OperationLike, r#type::TypeId, OperationRef},
///     pass::{create_external, ExternalPass},
///     Context,
/// };
///
/// #[repr(align(8))]
/// struct PassId;
///
/// static EXAMPLE_PASS: PassId = PassId;
///
/// let context = Context::new();
///
/// create_external(
///     |operation: OperationRef, _pass: ExternalPass| {
///         operation.dump();
///     },
///     TypeId::create(&EXAMPLE_PASS),
///     "name",
///     "argument",
///     "description",
///     "",
///     &[],
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn create_external<'c, T: RunExternalPass<'c>>(
    pass: T,
    pass_id: TypeId,
    name: &str,
    argument: &str,
    description: &str,
    op_name: &str,
    dependent_dialects: &[DialectHandle],
) -> Pass {
    unsafe {
        Pass::from_raw(mlirCreateExternalPass(
            pass_id.to_raw(),
            StringRef::new(name).to_raw(),
            StringRef::new(argument).to_raw(),
            StringRef::new(description).to_raw(),
            StringRef::new(op_name).to_raw(),
            dependent_dialects.len() as isize,
            dependent_dialects.as_ptr().cast_mut() as _,
            MlirExternalPassCallbacks {
                construct: Some(transmute::<*const (), unsafe extern "C" fn(*mut c_void)>(
                    callback_construct::<T> as *const (),
                )),
                destruct: Some(transmute::<*const (), unsafe extern "C" fn(*mut c_void)>(
                    callback_destruct::<T> as *const (),
                )),
                initialize: Some(transmute::<
                    *const (),
                    unsafe extern "C" fn(MlirContext, *mut c_void) -> MlirLogicalResult,
                >(callback_initialize::<T> as *const ())),
                run: Some(transmute::<
                    *const (),
                    unsafe extern "C" fn(MlirOperation, MlirExternalPass, *mut c_void),
                >(callback_run::<T> as *const ())),
                clone: Some(transmute::<
                    *const (),
                    unsafe extern "C" fn(*mut c_void) -> *mut c_void,
                >(callback_clone::<T> as *const ())),
            },
            Box::into_raw(Box::new(pass)) as _,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dialect::func,
        ir::{
            attribute::{StringAttribute, TypeAttribute},
            block::BlockLike,
            operation::OperationLike,
            r#type::FunctionType,
            Block, Identifier, Location, Module, Region, RegionLike,
        },
        pass::PassManager,
        test::create_test_context,
        Context,
    };

    #[repr(align(8))]
    struct PassId;

    fn create_module(context: &Context) -> Module {
        let location = Location::unknown(context);
        let module = Module::new(location);

        module.body().append_operation(func::func(
            context,
            StringAttribute::new(context, "foo"),
            TypeAttribute::new(FunctionType::new(context, &[], &[]).into()),
            {
                let block = Block::new(&[]);
                block.append_operation(func::r#return(&[], location));

                let region = Region::new();
                region.append_block(block);
                region
            },
            &[],
            location,
        ));
        module
    }

    #[test]
    fn external_pass() {
        static TEST_PASS: PassId = PassId;

        #[derive(Clone, Debug)]
        struct TestPass<'c> {
            context: &'c Context,
            value: i32,
        }

        impl<'c> RunExternalPass<'c> for TestPass<'c> {
            fn construct(&mut self) {
                assert_eq!(self.value, 10);
            }

            fn destruct(&mut self) {
                assert_eq!(self.value, 30);
            }

            fn initialize(&mut self, _context: ContextRef<'c>) {
                assert_eq!(self.value, 10);
                self.value = 20;
            }

            fn run(&mut self, operation: OperationRef<'c, '_>, _pass: ExternalPass<'_>) {
                assert_eq!(self.value, 20);
                self.value = 30;
                assert!(operation.verify());
                assert!(
                    operation
                        .region(0)
                        .expect("module has a body")
                        .first_block()
                        .expect("module has a body")
                        .first_operation()
                        .expect("body has a function")
                        .name()
                        == Identifier::new(self.context, "func.func")
                );
            }
        }

        impl TestPass<'_> {
            fn into_pass(self) -> Pass {
                create_external(
                    self,
                    TypeId::create(&TEST_PASS),
                    "test pass",
                    "test argument",
                    "a test pass",
                    "",
                    &[DialectHandle::func()],
                )
            }
        }

        let context = create_test_context();

        let mut module = create_module(&context);
        let pass_manager = PassManager::new(&context);

        let test_pass = TestPass {
            context: &context,
            value: 10,
        };
        pass_manager.add_pass(test_pass.into_pass());
        pass_manager.run(&mut module).unwrap();
    }

    #[test]
    fn external_fn_pass_failure() {
        static TEST_FN_PASS: PassId = PassId;

        let context = create_test_context();

        let mut module = create_module(&context);
        let pass_manager = PassManager::new(&context);

        pass_manager.add_pass(create_external(
            |operation: OperationRef, pass: ExternalPass| {
                assert!(operation.verify());
                assert!(
                    operation
                        .region(0)
                        .expect("module has a body")
                        .first_block()
                        .expect("module has a body")
                        .first_operation()
                        .expect("body has a function")
                        .name()
                        == Identifier::new(&context, "func.func")
                );
                pass.signal_failure();
            },
            TypeId::create(&TEST_FN_PASS),
            "test closure",
            "test argument",
            "test",
            "",
            &[DialectHandle::func()],
        ));
        assert!(pass_manager.run(&mut module).is_err());
    }
}
