use super::Context;
use core::{
    ffi::CStr,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};
use mlir_sys::{LLVMDisposeMessage, LLVMDisposeModule, LLVMModuleRef, LLVMPrintModuleToString};

/// An LLVM IR module.
#[derive(Debug)]
pub struct Module<'c> {
    raw: LLVMModuleRef,
    _context: PhantomData<&'c Context>,
}

impl Module<'_> {
    /// Creates a module from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: LLVMModuleRef) -> Self {
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
    pub unsafe fn from_option_raw(raw: LLVMModuleRef) -> Option<Self> {
        if raw.is_null() {
            None
        } else {
            Some(unsafe { Self::from_raw(raw) })
        }
    }

    /// Converts a module into a raw object.
    pub const fn to_raw(&self) -> LLVMModuleRef {
        self.raw
    }
}

impl Display for Module<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let string = unsafe { LLVMPrintModuleToString(self.raw) };
        let result = unsafe { CStr::from_ptr(string) }
            .to_str()
            .map_err(|_| fmt::Error)
            .and_then(|string| write!(formatter, "{string}"));

        unsafe { LLVMDisposeMessage(string) };

        result
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeModule(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlir_sys::LLVMModuleCreateWithNameInContext;

    #[test]
    fn display() {
        let context = Context::new();
        let module = unsafe {
            Module::from_raw(LLVMModuleCreateWithNameInContext(
                c"foo".as_ptr(),
                context.to_raw(),
            ))
        };

        insta::assert_snapshot!(module);
    }
}
