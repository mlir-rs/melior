//! Type IDs and allocators

mod allocator;

pub use allocator::Allocator;
use mlir_sys::{mlirTypeIDCreate, mlirTypeIDEqual, mlirTypeIDHashValue, MlirTypeID};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

/// A type ID.
#[derive(Clone, Copy, Debug)]
pub struct TypeId<'c> {
    raw: MlirTypeID,
    _owner: PhantomData<&'c ()>,
}

impl TypeId<'_> {
    /// Creates a type ID from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirTypeID) -> Self {
        Self {
            raw,
            _owner: PhantomData,
        }
    }

    /// Converts a type ID into a raw object.
    pub const fn to_raw(self) -> MlirTypeID {
        self.raw
    }

    /// Creates a type ID from an 8-byte aligned reference.
    ///
    /// # Panics
    ///
    /// This function will panic if the given reference is not 8-byte aligned.
    // TODO Return a result instead of using assertion.
    pub fn create<T>(reference: &T) -> Self {
        let ptr = reference as *const _ as *const std::ffi::c_void;

        assert_eq!(
            ptr.align_offset(8),
            0,
            "type ID pointer must be 8-byte aligned"
        );

        unsafe { Self::from_raw(mlirTypeIDCreate(ptr)) }
    }
}

impl PartialEq for TypeId<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirTypeIDEqual(self.raw, other.raw) }
    }
}

impl Eq for TypeId<'_> {}

impl Hash for TypeId<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        unsafe {
            mlirTypeIDHashValue(self.raw).hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_from_reference() {
        static VALUE: u64 = 0;

        TypeId::create(&VALUE);
    }

    #[test]
    #[should_panic]
    fn reject_invalid_alignment() {
        let values: (u32, u8) = Default::default();

        TypeId::create(&values.1);
    }
}
