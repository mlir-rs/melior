use mlir_sys::{LLVMContextCreate, LLVMContextDispose, LLVMContextRef};

/// An LLVM context.
#[derive(Debug)]
pub struct Context {
    raw: LLVMContextRef,
}

impl Context {
    /// Creates an LLVM context.
    pub fn new() -> Self {
        Self {
            raw: unsafe { LLVMContextCreate() },
        }
    }

    /// Converts an LLVM context into a raw object.
    pub const fn to_raw(&self) -> LLVMContextRef {
        self.raw
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { LLVMContextDispose(self.raw) };
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Context::new();
    }
}
