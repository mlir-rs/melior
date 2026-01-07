use crate::{
    context::{Context, ContextRef},
    utility::print_callback,
};
use mlir_sys::{
    MlirAffineMap, mlirAffineMapDump, mlirAffineMapEqual, mlirAffineMapGetContext,
    mlirAffineMapPrint,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

/// An affine map.
#[derive(Clone, Copy)]
pub struct AffineMap<'c> {
    raw: MlirAffineMap,
    _context: PhantomData<&'c Context>,
}

impl<'c> AffineMap<'c> {
    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineMapGetContext(self.raw)) }
    }

    /// Dumps an affine map.
    pub fn dump(&self) {
        unsafe { mlirAffineMapDump(self.raw) }
    }

    /// Creates an affine map from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirAffineMap) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }
}

impl PartialEq for AffineMap<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineMapEqual(self.raw, other.raw) }
    }
}

impl Eq for AffineMap<'_> {}

impl Display for AffineMap<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirAffineMapPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for AffineMap<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
