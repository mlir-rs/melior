use crate::{
    context::{Context, ContextRef},
    ir::{Attribute, AttributeLike, Identifier},
    string_ref::StringRef,
    utility::print_callback,
};
use mlir_sys::{
    MlirLocation, mlirLocationCallSiteGet, mlirLocationCallSiteGetCallee,
    mlirLocationCallSiteGetCaller, mlirLocationEqual, mlirLocationFileLineColGet,
    mlirLocationFileLineColRangeGet, mlirLocationFileLineColRangeGetEndColumn,
    mlirLocationFileLineColRangeGetEndLine, mlirLocationFileLineColRangeGetFilename,
    mlirLocationFileLineColRangeGetStartColumn, mlirLocationFileLineColRangeGetStartLine,
    mlirLocationFromAttribute, mlirLocationFusedGet, mlirLocationFusedGetLocations,
    mlirLocationFusedGetMetadata, mlirLocationFusedGetNumLocations, mlirLocationGetAttribute,
    mlirLocationGetContext, mlirLocationIsACallSite, mlirLocationIsAFileLineColRange,
    mlirLocationIsAFused, mlirLocationIsAName, mlirLocationNameGet, mlirLocationNameGetChildLoc,
    mlirLocationNameGetName, mlirLocationPrint, mlirLocationUnknownGet,
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

    /// Creates a file/line/column range location.
    pub fn file_line_col_range(
        context: &'c Context,
        filename: &str,
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirLocationFileLineColRangeGet(
                context.to_raw(),
                StringRef::new(filename).to_raw(),
                start_line as u32,
                start_col as u32,
                end_line as u32,
                end_col as u32,
            ))
        }
    }

    /// Creates an unknown location.
    pub fn unknown(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirLocationUnknownGet(context.to_raw())) }
    }

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirLocationGetContext(self.raw)) }
    }

    /// Returns `true` if this location is a call site location.
    pub fn is_call_site(&self) -> bool {
        unsafe { mlirLocationIsACallSite(self.raw) }
    }

    /// Returns `true` if this location is a file/line/column range location.
    pub fn is_file_line_col_range(&self) -> bool {
        unsafe { mlirLocationIsAFileLineColRange(self.raw) }
    }

    /// Returns `true` if this location is a fused location.
    pub fn is_fused(&self) -> bool {
        unsafe { mlirLocationIsAFused(self.raw) }
    }

    /// Returns `true` if this location is a name location.
    pub fn is_name(&self) -> bool {
        unsafe { mlirLocationIsAName(self.raw) }
    }

    /// Returns the callee of a call site location.
    pub fn call_site_callee(&self) -> Self {
        unsafe { Self::from_raw(mlirLocationCallSiteGetCallee(self.raw)) }
    }

    /// Returns the caller of a call site location.
    pub fn call_site_caller(&self) -> Self {
        unsafe { Self::from_raw(mlirLocationCallSiteGetCaller(self.raw)) }
    }

    /// Returns the filename of a file/line/column range location.
    pub fn file_line_col_range_filename(&self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirLocationFileLineColRangeGetFilename(self.raw)) }
    }

    /// Returns the start line of a file/line/column range location.
    pub fn file_line_col_range_start_line(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetStartLine(self.raw) as usize }
    }

    /// Returns the start column of a file/line/column range location.
    pub fn file_line_col_range_start_column(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetStartColumn(self.raw) as usize }
    }

    /// Returns the end line of a file/line/column range location.
    pub fn file_line_col_range_end_line(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetEndLine(self.raw) as usize }
    }

    /// Returns the end column of a file/line/column range location.
    pub fn file_line_col_range_end_column(&self) -> usize {
        unsafe { mlirLocationFileLineColRangeGetEndColumn(self.raw) as usize }
    }

    /// Returns the number of locations in a fused location.
    pub fn fused_num_locations(&self) -> usize {
        unsafe { mlirLocationFusedGetNumLocations(self.raw) as usize }
    }

    /// Returns the location at `index` in a fused location.
    pub fn fused_location(&self, index: usize) -> Self {
        let count = self.fused_num_locations();
        let mut locs =
            vec![
                unsafe { Self::from_raw(mlirLocationUnknownGet(mlirLocationGetContext(self.raw))) };
                count
            ];
        unsafe {
            mlirLocationFusedGetLocations(self.raw, locs.as_mut_ptr() as *mut _);
            locs[index]
        }
    }

    /// Returns the metadata attribute of a fused location.
    pub fn fused_metadata(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirLocationFusedGetMetadata(self.raw)) }
    }

    /// Returns the name identifier of a name location.
    pub fn name_value(&self) -> Identifier<'c> {
        unsafe { Identifier::from_raw(mlirLocationNameGetName(self.raw)) }
    }

    /// Returns the child location of a name location.
    pub fn name_child_location(&self) -> Self {
        unsafe { Self::from_raw(mlirLocationNameGetChildLoc(self.raw)) }
    }

    /// Converts this location to an attribute.
    pub fn to_attribute(&self) -> Attribute<'c> {
        unsafe { Attribute::from_raw(mlirLocationGetAttribute(self.raw)) }
    }

    /// Creates a location from an attribute.
    pub fn from_attribute(attribute: Attribute<'c>) -> Self {
        unsafe { Self::from_raw(mlirLocationFromAttribute(attribute.to_raw())) }
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

    #[test]
    fn file_line_col_range() {
        let context = Context::new();

        Location::file_line_col_range(&context, "foo.mlir", 1, 0, 1, 10);
    }

    #[test]
    fn is_call_site() {
        let context = Context::new();

        let loc = Location::call_site(Location::unknown(&context), Location::unknown(&context));
        assert!(loc.is_call_site());
        assert!(!Location::unknown(&context).is_call_site());
    }

    #[test]
    fn is_file_line_col_range() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 1, 0, 1, 10);
        assert!(loc.is_file_line_col_range());
        assert!(!Location::unknown(&context).is_file_line_col_range());
    }

    #[test]
    fn is_fused() {
        let context = Context::new();

        let loc = Location::fused(
            &context,
            &[Location::unknown(&context)],
            Attribute::parse(&context, "42").unwrap(),
        );
        assert!(loc.is_fused());
        assert!(!Location::unknown(&context).is_fused());
    }

    #[test]
    fn is_name() {
        let context = Context::new();

        let loc = Location::name(&context, "foo", Location::unknown(&context));
        assert!(loc.is_name());
        assert!(!Location::unknown(&context).is_name());
    }

    #[test]
    fn call_site_callee() {
        let context = Context::new();

        let callee = Location::new(&context, "callee.mlir", 1, 1);
        let caller = Location::new(&context, "caller.mlir", 2, 2);
        let loc = Location::call_site(callee, caller);
        assert_eq!(loc.call_site_callee(), callee);
    }

    #[test]
    fn call_site_caller() {
        let context = Context::new();

        let callee = Location::new(&context, "callee.mlir", 1, 1);
        let caller = Location::new(&context, "caller.mlir", 2, 2);
        let loc = Location::call_site(callee, caller);
        assert_eq!(loc.call_site_caller(), caller);
    }

    #[test]
    fn file_line_col_range_filename() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 1, 0, 1, 10);
        assert_eq!(
            loc.file_line_col_range_filename().as_string_ref().as_str(),
            Ok("foo.mlir")
        );
    }

    #[test]
    fn file_line_col_range_start_line() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 3, 4, 5, 6);
        assert_eq!(loc.file_line_col_range_start_line(), 3);
    }

    #[test]
    fn file_line_col_range_start_column() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 3, 4, 5, 6);
        assert_eq!(loc.file_line_col_range_start_column(), 4);
    }

    #[test]
    fn file_line_col_range_end_line() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 3, 4, 5, 6);
        assert_eq!(loc.file_line_col_range_end_line(), 5);
    }

    #[test]
    fn file_line_col_range_end_column() {
        let context = Context::new();

        let loc = Location::file_line_col_range(&context, "foo.mlir", 3, 4, 5, 6);
        assert_eq!(loc.file_line_col_range_end_column(), 6);
    }

    #[test]
    fn fused_num_locations() {
        let context = Context::new();

        let loc = Location::fused(
            &context,
            &[
                Location::new(&context, "foo", 1, 1),
                Location::new(&context, "bar", 2, 2),
            ],
            Attribute::parse(&context, "42").unwrap(),
        );
        assert_eq!(loc.fused_num_locations(), 2);
    }

    #[test]
    fn fused_location() {
        let context = Context::new();

        let a = Location::new(&context, "foo", 1, 1);
        let b = Location::new(&context, "bar", 2, 2);
        let loc = Location::fused(&context, &[a, b], Attribute::parse(&context, "42").unwrap());
        assert_eq!(loc.fused_location(0), a);
        assert_eq!(loc.fused_location(1), b);
    }

    #[test]
    fn fused_metadata() {
        let context = Context::new();

        let attr = Attribute::parse(&context, "42").unwrap();
        let loc = Location::fused(&context, &[Location::unknown(&context)], attr);
        assert_eq!(loc.fused_metadata(), attr);
    }

    #[test]
    fn name_value() {
        let context = Context::new();

        let loc = Location::name(&context, "my_name", Location::unknown(&context));
        assert_eq!(loc.name_value().as_string_ref().as_str(), Ok("my_name"));
    }

    #[test]
    fn name_child_location() {
        let context = Context::new();

        let child = Location::unknown(&context);
        let loc = Location::name(&context, "my_name", child);
        assert_eq!(loc.name_child_location(), child);
    }

    #[test]
    fn to_attribute() {
        let context = Context::new();

        Location::unknown(&context).to_attribute();
    }

    #[test]
    fn from_attribute() {
        let context = Context::new();

        let loc = Location::unknown(&context);
        let attr = loc.to_attribute();
        assert_eq!(Location::from_attribute(attr), loc);
    }
}
