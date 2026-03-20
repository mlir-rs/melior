use mlir_sys::{MlirLlvmThreadPool, mlirLlvmThreadPoolCreate, mlirLlvmThreadPoolDestroy};

/// An LLVM thread pool.
pub struct ThreadPool {
    raw: MlirLlvmThreadPool,
}

impl ThreadPool {
    /// Creates a thread pool.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirLlvmThreadPoolCreate() },
        }
    }

    /// Converts a thread pool into a raw object.
    pub const fn to_raw(&self) -> MlirLlvmThreadPool {
        self.raw
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        unsafe { mlirLlvmThreadPoolDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn new() {
        ThreadPool::new();
    }

    #[test]
    fn set_thread_pool() {
        let pool = ThreadPool::new();
        let context = Context::new();

        unsafe { context.set_thread_pool(&pool) };

        assert!(context.thread_count() > 0);
    }
}
