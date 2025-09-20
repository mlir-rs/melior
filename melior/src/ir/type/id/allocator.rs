use super::TypeId;
use mlir_sys::{
    mlirTypeIDAllocatorAllocateTypeID, mlirTypeIDAllocatorCreate, mlirTypeIDAllocatorDestroy,
    MlirTypeIDAllocator,
};

/// A type ID allocator.
#[derive(Debug)]
pub struct Allocator {
    raw: MlirTypeIDAllocator,
}

impl Allocator {
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirTypeIDAllocatorCreate() },
        }
    }

    pub fn allocate_type_id(&mut self) -> TypeId<'_> {
        unsafe { TypeId::from_raw(mlirTypeIDAllocatorAllocateTypeID(self.raw)) }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        unsafe { mlirTypeIDAllocatorDestroy(self.raw) }
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        Allocator::new();
    }

    #[test]
    fn allocate_type_id() {
        Allocator::new().allocate_type_id();
    }
}
