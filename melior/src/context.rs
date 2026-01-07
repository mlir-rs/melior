use crate::{
    diagnostic::{Diagnostic, DiagnosticHandlerId},
    dialect::{Dialect, DialectRegistry},
    logical_result::LogicalResult,
    string_ref::StringRef,
};
use mlir_sys::{
    MlirContext, MlirDiagnostic, MlirLogicalResult, mlirContextAppendDialectRegistry,
    mlirContextAttachDiagnosticHandler, mlirContextCreate, mlirContextDestroy,
    mlirContextDetachDiagnosticHandler, mlirContextEnableMultithreading, mlirContextEqual,
    mlirContextGetAllowUnregisteredDialects, mlirContextGetNumLoadedDialects,
    mlirContextGetNumRegisteredDialects, mlirContextGetOrLoadDialect,
    mlirContextIsRegisteredOperation, mlirContextLoadAllAvailableDialects,
    mlirContextSetAllowUnregisteredDialects,
};
use std::{ffi::c_void, marker::PhantomData, mem::transmute};

/// A context of IR, dialects, and passes.
///
/// Contexts own various objects, such as types, locations, and dialect
/// instances.
#[derive(Debug)]
pub struct Context {
    raw: MlirContext,
}

impl Context {
    /// Creates a context.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirContextCreate() },
        }
    }

    /// Returns a number of registered dialects.
    pub fn registered_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumRegisteredDialects(self.raw) as usize }
    }

    /// Returns a number of loaded dialects.
    pub fn loaded_dialect_count(&self) -> usize {
        unsafe { mlirContextGetNumLoadedDialects(self.raw) as usize }
    }

    /// Returns or loads a dialect.
    pub fn get_or_load_dialect<'c>(&'c self, name: &str) -> Dialect<'c> {
        let name = StringRef::new(name);

        unsafe { Dialect::from_raw(mlirContextGetOrLoadDialect(self.raw, name.to_raw())) }
    }

    /// Appends a dialect registry.
    pub fn append_dialect_registry(&self, registry: &DialectRegistry) {
        unsafe { mlirContextAppendDialectRegistry(self.raw, registry.to_raw()) }
    }

    /// Loads all available dialects.
    pub fn load_all_available_dialects(&self) {
        unsafe { mlirContextLoadAllAvailableDialects(self.raw) }
    }

    /// Enables multi-threading.
    pub fn enable_multi_threading(&self, enabled: bool) {
        unsafe { mlirContextEnableMultithreading(self.raw, enabled) }
    }

    /// Returns `true` if unregistered dialects are allowed.
    pub fn allow_unregistered_dialects(&self) -> bool {
        unsafe { mlirContextGetAllowUnregisteredDialects(self.raw) }
    }

    /// Sets if unregistered dialects are allowed.
    pub fn set_allow_unregistered_dialects(&self, allowed: bool) {
        unsafe { mlirContextSetAllowUnregisteredDialects(self.raw, allowed) }
    }

    /// Returns `true` if a given operation is registered in a context.
    pub fn is_registered_operation(&self, name: &str) -> bool {
        let name = StringRef::new(name);

        unsafe { mlirContextIsRegisteredOperation(self.raw, name.to_raw()) }
    }

    /// Converts a context into a raw object.
    pub const fn to_raw(&self) -> MlirContext {
        self.raw
    }

    /// Attaches a diagnostic handler.
    pub fn attach_diagnostic_handler<F: FnMut(Diagnostic) -> bool>(
        &self,
        handler: F,
    ) -> DiagnosticHandlerId {
        unsafe extern "C" fn handle<F: FnMut(Diagnostic) -> bool>(
            diagnostic: MlirDiagnostic,
            user_data: *mut c_void,
        ) -> MlirLogicalResult {
            LogicalResult::from(unsafe {
                (*(user_data as *mut F))(Diagnostic::from_raw(diagnostic))
            })
            .to_raw()
        }

        unsafe extern "C" fn destroy<F: FnMut(Diagnostic) -> bool>(user_data: *mut c_void) {
            unsafe {
                drop(Box::from_raw(user_data as *mut F));
            }
        }

        unsafe {
            DiagnosticHandlerId::from_raw(mlirContextAttachDiagnosticHandler(
                self.to_raw(),
                Some(handle::<F>),
                Box::into_raw(Box::new(handler)) as *mut _,
                Some(destroy::<F>),
            ))
        }
    }

    /// Detaches a diagnostic handler.
    pub fn detach_diagnostic_handler(&self, id: DiagnosticHandlerId) {
        unsafe { mlirContextDetachDiagnosticHandler(self.to_raw(), id.to_raw()) }
    }

    pub(crate) fn to_ref(&self) -> ContextRef<'_> {
        unsafe { ContextRef::from_raw(self.to_raw()) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { mlirContextDestroy(self.raw) };
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}

impl<'a> PartialEq<ContextRef<'a>> for Context {
    fn eq(&self, &other: &ContextRef<'a>) -> bool {
        self.to_ref() == other
    }
}

impl Eq for Context {}

/// A reference to a context.
#[derive(Clone, Copy, Debug)]
pub struct ContextRef<'c> {
    raw: MlirContext,
    _reference: PhantomData<&'c Context>,
}

impl<'c> ContextRef<'c> {
    /// Creates a context reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirContext) -> Self {
        Self {
            raw,
            _reference: Default::default(),
        }
    }

    /// Returns a context.
    ///
    /// This function is different from `deref` because the correct lifetime is
    /// kept for the return type.
    ///
    /// # Safety
    ///
    /// The returned reference is safe to use only in the lifetime scope of the
    /// context reference.
    pub unsafe fn to_ref(&self) -> &'c Context {
        // As we can't deref ContextRef<'a> into `&'a Context`, we forcibly cast its
        // lifetime here to extend it from the lifetime of `ObjectRef<'a>` itself into
        // `'a`.
        unsafe { transmute(self) }
    }
}

impl PartialEq for ContextRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirContextEqual(self.raw, other.raw) }
    }
}

impl PartialEq<Context> for ContextRef<'_> {
    fn eq(&self, other: &Context) -> bool {
        self == &other.to_ref()
    }
}

impl Eq for ContextRef<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }

    #[test]
    fn registered_dialect_count() {
        let context = Context::new();

        assert_eq!(context.registered_dialect_count(), 1);
    }

    #[test]
    fn loaded_dialect_count() {
        let context = Context::new();

        assert_eq!(context.loaded_dialect_count(), 1);
    }

    #[test]
    fn append_dialect_registry() {
        let context = Context::new();

        context.append_dialect_registry(&DialectRegistry::new());
    }

    #[test]
    fn is_registered_operation() {
        let context = Context::new();

        assert!(context.is_registered_operation("builtin.module"));
    }

    #[test]
    fn is_not_registered_operation() {
        let context = Context::new();

        assert!(!context.is_registered_operation("func.func"));
    }

    #[test]
    fn enable_multi_threading() {
        let context = Context::new();

        context.enable_multi_threading(true);
    }

    #[test]
    fn disable_multi_threading() {
        let context = Context::new();

        context.enable_multi_threading(false);
    }

    #[test]
    fn allow_unregistered_dialects() {
        let context = Context::new();

        assert!(!context.allow_unregistered_dialects());
    }

    #[test]
    fn set_allow_unregistered_dialects() {
        let context = Context::new();

        context.set_allow_unregistered_dialects(true);

        assert!(context.allow_unregistered_dialects());
    }

    #[test]
    fn attach_and_detach_diagnostic_handler() {
        let context = Context::new();

        let id = context.attach_diagnostic_handler(|diagnostic| {
            println!("{diagnostic}");
            true
        });

        context.detach_diagnostic_handler(id);
    }

    #[test]
    fn compare_contexts() {
        let one = Context::new();
        let other = Context::new();

        assert_eq!(&one, &one);
        assert_ne!(&one, &other);
        assert_ne!(&other, &one);
        assert_eq!(&other, &other);
    }

    #[test]
    fn compare_context_refs() {
        let one = Context::new();
        let other = Context::new();

        let one_ref = one.to_ref();
        let other_ref = other.to_ref();

        assert_eq!(&one, &one_ref);
        assert_eq!(&one_ref, &one);

        assert_eq!(&other, &other_ref);
        assert_eq!(&other_ref, &other);

        assert_ne!(&one, &other_ref);
        assert_ne!(&other_ref, &one);

        assert_ne!(&other, &one_ref);
        assert_ne!(&one_ref, &other);
    }

    #[test]
    fn context_to_ref() {
        let ctx = Context::new();
        let ctx_ref = ctx.to_ref();
        let ctx_ref_to_ref: &Context = unsafe { ctx_ref.to_ref() };

        assert_eq!(&ctx_ref, ctx_ref_to_ref);
    }
}
