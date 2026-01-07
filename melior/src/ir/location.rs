use crate::{
    context::{Context, ContextRef},
    ir::{Attribute, AttributeLike},
    string_ref::StringRef,
    utility::print_callback,
};
use mlir_sys::{
    MlirLocation, mlirLocationCallSiteGet, mlirLocationEqual, mlirLocationFileLineColGet,
    mlirLocationFusedGet, mlirLocationGetContext, mlirLocationNameGet, mlirLocationPrint,
    mlirLocationUnknownGet,
};
use std::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};

/// A location
#[derive(Clone, Copy, Debug)]
pub struct Location<'c> {
    raw: MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    /// Creates a location with a filename and line and column numbers.
    pub fn new(context: &'c Context, filename: &str, line: usize, column: usize) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColGet(
                context.to_raw(),
                StringRef::new(filename).to_raw(),
                line as u32,
                column as u32,
            ))
        }
    }

    /// Creates a fused location.
    pub fn fused(context: &'c Context, locations: &[Self], attribute: Attribute) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFusedGet(
                context.to_raw(),
                locations.len() as isize,
                locations as *const _ as *const _,
                attribute.to_raw(),
            ))
        }
    }

    /// Creates a name location.
    pub fn name(context: &'c Context, name: &str, child: Location) -> Self {
        unsafe {
            Self::from_raw(mlirLocationNameGet(
                context.to_raw(),
                StringRef::new(name).to_raw(),
                child.to_raw(),
            ))
        }
    }

    /// Creates a call site location.
    pub fn call_site(callee: Location, caller: Location) -> Self {
        unsafe { Self::from_raw(mlirLocationCallSiteGet(callee.to_raw(), caller.to_raw())) }
    }

    /// Creates an unknown location.
    pub fn unknown(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(context.to_raw())) }
    }

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
    }

    /// Creates a location from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirLocation) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts a location into a raw object.
    pub const fn to_raw(self) -> MlirLocation {
        self.raw
    }
}

impl PartialEq for Location<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirLocationEqual(self.raw, other.raw) }
    }
}

impl Display for Location<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirLocationPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn new() {
        Location::new(&Context::new(), "foo", 42, 42);
    }

    #[test]
    fn fused() {
        let context = Context::new();

        Location::fused(
            &context,
            &[
                Location::new(&Context::new(), "foo", 1, 1),
                Location::new(&Context::new(), "foo", 2, 2),
            ],
            Attribute::parse(&context, "42").unwrap(),
        );
    }

    #[test]
    fn name() {
        let context = Context::new();

        Location::name(&context, "foo", Location::unknown(&context));
    }

    #[test]
    fn call_site() {
        let context = Context::new();

        Location::call_site(Location::unknown(&context), Location::unknown(&context));
    }

    #[test]
    fn unknown() {
        Location::unknown(&Context::new());
    }

    #[test]
    fn context() {
        Location::new(&Context::new(), "foo", 42, 42).context();
    }

    #[test]
    fn equal() {
        let context = Context::new();

        assert_eq!(Location::unknown(&context), Location::unknown(&context));
        assert_eq!(
            Location::new(&context, "foo", 42, 42),
            Location::new(&context, "foo", 42, 42),
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();

        assert_ne!(
            Location::new(&context, "foo", 42, 42),
            Location::unknown(&context)
        );
    }

    #[test]
    fn display() {
        let context = Context::new();

        assert_eq!(Location::unknown(&context).to_string(), "loc(unknown)");
        assert_eq!(
            Location::new(&context, "foo", 42, 42).to_string(),
            "loc(\"foo\":42:42)"
        );
    }
}
