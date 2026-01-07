use mlir_sys::{MlirStringRef, mlirStringRefEqual};
use std::{
    ffi::CStr,
    marker::PhantomData,
    slice,
    str::{self, Utf8Error},
};

/// A string reference.
#[derive(Clone, Copy, Debug)]
pub struct StringRef<'a> {
    raw: MlirStringRef,
    _parent: PhantomData<&'a str>,
}

impl<'a> StringRef<'a> {
    /// Creates a string reference.
    pub fn new(string: &'a str) -> Self {
        let string = MlirStringRef {
            data: string.as_bytes().as_ptr() as *const _,
            length: string.len(),
        };

        unsafe { Self::from_raw(string) }
    }

    /// Converts a C-style string into a string reference.
    pub fn from_c_str(string: &'a CStr) -> Self {
        let string = MlirStringRef {
            data: string.as_ptr(),
            length: string.to_bytes_with_nul().len() - 1,
        };

        unsafe { Self::from_raw(string) }
    }

    /// Converts a string reference into a `str`.
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        unsafe {
            let bytes = slice::from_raw_parts(self.raw.data as *mut u8, self.raw.length);

            str::from_utf8(if bytes[bytes.len() - 1] == 0 {
                &bytes[..bytes.len() - 1]
            } else {
                bytes
            })
        }
    }

    /// Converts a string reference into a raw object.
    pub const fn to_raw(self) -> MlirStringRef {
        self.raw
    }

    /// Creates a string reference from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(string: MlirStringRef) -> Self {
        Self {
            raw: string,
            _parent: Default::default(),
        }
    }
}

impl PartialEq for StringRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirStringRefEqual(self.raw, other.raw) }
    }
}

impl Eq for StringRef<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal() {
        assert_eq!(StringRef::new("foo"), StringRef::new("foo"));
    }

    #[test]
    fn equal_str() {
        assert_eq!(StringRef::new("foo").as_str().unwrap(), "foo");
    }

    #[test]
    fn not_equal() {
        assert_ne!(StringRef::new("foo"), StringRef::new("bar"));
    }
}
