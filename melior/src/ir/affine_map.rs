use crate::{
    Error,
    context::{Context, ContextRef},
    ir::AffineExpr,
    utility::print_callback,
};
use mlir_sys::{
    MlirAffineMap, mlirAffineMapConstantGet, mlirAffineMapDump, mlirAffineMapEmptyGet,
    mlirAffineMapEqual, mlirAffineMapGet, mlirAffineMapGetContext, mlirAffineMapGetMajorSubMap,
    mlirAffineMapGetMinorSubMap, mlirAffineMapGetNumDims, mlirAffineMapGetNumInputs,
    mlirAffineMapGetNumResults, mlirAffineMapGetNumSymbols, mlirAffineMapGetResult,
    mlirAffineMapGetSingleConstantResult, mlirAffineMapGetSubMap, mlirAffineMapIsEmpty,
    mlirAffineMapIsIdentity, mlirAffineMapIsMinorIdentity, mlirAffineMapIsPermutation,
    mlirAffineMapIsProjectedPermutation, mlirAffineMapIsSingleConstant,
    mlirAffineMapMinorIdentityGet, mlirAffineMapMultiDimIdentityGet, mlirAffineMapPermutationGet,
    mlirAffineMapPrint, mlirAffineMapReplace, mlirAffineMapZeroResultGet,
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
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Creates an empty affine map (no dimensions, symbols, or results).
    pub fn empty(context: &'c Context) -> Self {
        unsafe { Self::from_raw(mlirAffineMapEmptyGet(context.to_raw())) }
    }

    /// Creates a zero-result affine map with the given number of dimensions and
    /// symbols.
    pub fn zero_result(context: &'c Context, dims: usize, symbols: usize) -> Self {
        unsafe {
            Self::from_raw(mlirAffineMapZeroResultGet(
                context.to_raw(),
                dims as isize,
                symbols as isize,
            ))
        }
    }

    /// Creates an affine map with results defined by the given affine
    /// expressions.
    pub fn new(
        context: &'c Context,
        dims: usize,
        symbols: usize,
        exprs: &[AffineExpr<'c>],
    ) -> Self {
        let mut raws: Vec<_> = exprs.iter().map(|e| e.to_raw()).collect();
        unsafe {
            Self::from_raw(mlirAffineMapGet(
                context.to_raw(),
                dims as isize,
                symbols as isize,
                raws.len() as isize,
                raws.as_mut_ptr(),
            ))
        }
    }

    /// Creates a single constant result affine map.
    pub fn constant(context: &'c Context, value: i64) -> Self {
        unsafe { Self::from_raw(mlirAffineMapConstantGet(context.to_raw(), value)) }
    }

    /// Creates an identity affine map with the given number of dimensions.
    pub fn multi_dim_identity(context: &'c Context, dims: usize) -> Self {
        unsafe {
            Self::from_raw(mlirAffineMapMultiDimIdentityGet(
                context.to_raw(),
                dims as isize,
            ))
        }
    }

    /// Creates a minor identity affine map with the given number of dimensions
    /// and results.
    pub fn minor_identity(context: &'c Context, dims: usize, results: usize) -> Self {
        unsafe {
            Self::from_raw(mlirAffineMapMinorIdentityGet(
                context.to_raw(),
                dims as isize,
                results as isize,
            ))
        }
    }

    /// Creates an affine map representing a permutation.
    pub fn permutation(context: &'c Context, permutation: &[u32]) -> Self {
        let mut perm: Vec<_> = permutation
            .iter()
            .map(|&x| x as std::os::raw::c_uint)
            .collect();
        unsafe {
            Self::from_raw(mlirAffineMapPermutationGet(
                context.to_raw(),
                perm.len() as isize,
                perm.as_mut_ptr(),
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// Returns a context.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineMapGetContext(self.raw)) }
    }

    /// Returns the number of dimensions.
    pub fn dim_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumDims(self.raw) as usize }
    }

    /// Returns the number of symbols.
    pub fn symbol_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumSymbols(self.raw) as usize }
    }

    /// Returns the number of results.
    pub fn result_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumResults(self.raw) as usize }
    }

    /// Returns the number of inputs (dimensions + symbols).
    pub fn input_count(&self) -> usize {
        unsafe { mlirAffineMapGetNumInputs(self.raw) as usize }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Returns the result at the given index.
    pub fn result(&self, index: usize) -> Result<AffineExpr<'c>, Error> {
        if index < self.result_count() {
            Ok(unsafe { AffineExpr::from_raw(mlirAffineMapGetResult(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "affine map result",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns the single constant result. Only valid when
    /// `is_single_constant()` is true.
    pub fn single_constant_result(&self) -> i64 {
        unsafe { mlirAffineMapGetSingleConstantResult(self.raw) }
    }

    /// Returns the sub-map consisting of the most major `n` results.
    pub fn major_sub_map(&self, n: usize) -> Self {
        unsafe { Self::from_raw(mlirAffineMapGetMajorSubMap(self.raw, n as isize)) }
    }

    /// Returns the sub-map consisting of the most minor `n` results.
    pub fn minor_sub_map(&self, n: usize) -> Self {
        unsafe { Self::from_raw(mlirAffineMapGetMinorSubMap(self.raw, n as isize)) }
    }

    /// Returns the sub-map at the given result positions.
    pub fn sub_map(&self, positions: &[usize]) -> Self {
        let mut pos: Vec<isize> = positions.iter().map(|&p| p as isize).collect();
        unsafe {
            Self::from_raw(mlirAffineMapGetSubMap(
                self.raw,
                pos.len() as isize,
                pos.as_mut_ptr(),
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Predicates
    // -----------------------------------------------------------------------

    /// Returns `true` if the affine map is empty (no dims, symbols, or
    /// results).
    pub fn is_empty(&self) -> bool {
        unsafe { mlirAffineMapIsEmpty(self.raw) }
    }

    /// Returns `true` if the affine map is an identity map.
    pub fn is_identity(&self) -> bool {
        unsafe { mlirAffineMapIsIdentity(self.raw) }
    }

    /// Returns `true` if the affine map is a minor identity map.
    pub fn is_minor_identity(&self) -> bool {
        unsafe { mlirAffineMapIsMinorIdentity(self.raw) }
    }

    /// Returns `true` if the affine map is a permutation.
    pub fn is_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsPermutation(self.raw) }
    }

    /// Returns `true` if the affine map is a projected permutation.
    pub fn is_projected_permutation(&self) -> bool {
        unsafe { mlirAffineMapIsProjectedPermutation(self.raw) }
    }

    /// Returns `true` if the affine map has a single constant result.
    pub fn is_single_constant(&self) -> bool {
        unsafe { mlirAffineMapIsSingleConstant(self.raw) }
    }

    // -----------------------------------------------------------------------
    // Transforms
    // -----------------------------------------------------------------------

    /// Replaces all occurrences of `expr` with `replacement` in the affine
    /// map, producing a new map with `dims` dimensions and `symbols` symbols.
    pub fn replace(
        &self,
        expr: AffineExpr<'c>,
        replacement: AffineExpr<'c>,
        dims: usize,
        symbols: usize,
    ) -> Self {
        unsafe {
            Self::from_raw(mlirAffineMapReplace(
                self.raw,
                expr.to_raw(),
                replacement.to_raw(),
                dims as isize,
                symbols as isize,
            ))
        }
    }

    // TODO: mlirAffineMapCompressUnusedSymbols — complex C API with mutable
    // output arrays and a callback; deferred.

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

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

    /// Converts an affine map into a raw object.
    pub const fn to_raw(self) -> MlirAffineMap {
        self.raw
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    use pretty_assertions::assert_eq;

    #[test]
    fn empty() {
        let context = Context::new();
        let map = AffineMap::empty(&context);
        assert!(map.is_empty());
        assert_eq!(map.dim_count(), 0);
        assert_eq!(map.symbol_count(), 0);
        assert_eq!(map.result_count(), 0);
    }

    #[test]
    fn zero_result() {
        let context = Context::new();
        let map = AffineMap::zero_result(&context, 2, 1);
        assert_eq!(map.dim_count(), 2);
        assert_eq!(map.symbol_count(), 1);
        assert_eq!(map.result_count(), 0);
    }

    #[test]
    fn new_with_exprs() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let d1 = AffineExpr::dim(&context, 1);
        let map = AffineMap::new(&context, 2, 0, &[d0, d1]);
        assert_eq!(map.dim_count(), 2);
        assert_eq!(map.result_count(), 2);
    }

    #[test]
    fn new_empty_exprs() {
        let context = Context::new();
        let map = AffineMap::new(&context, 0, 0, &[]);
        assert!(map.is_empty());
    }

    #[test]
    fn constant() {
        let context = Context::new();
        let map = AffineMap::constant(&context, 42);
        assert!(map.is_single_constant());
        assert_eq!(map.single_constant_result(), 42);
    }

    #[test]
    fn multi_dim_identity() {
        let context = Context::new();
        let map = AffineMap::multi_dim_identity(&context, 3);
        assert!(map.is_identity());
        assert_eq!(map.dim_count(), 3);
        assert_eq!(map.result_count(), 3);
    }

    #[test]
    fn minor_identity() {
        let context = Context::new();
        let map = AffineMap::minor_identity(&context, 3, 2);
        assert!(map.is_minor_identity());
        assert_eq!(map.dim_count(), 3);
        assert_eq!(map.result_count(), 2);
    }

    #[test]
    fn permutation() {
        let context = Context::new();
        let map = AffineMap::permutation(&context, &[1, 2, 0]);
        assert!(map.is_permutation());
        assert_eq!(map.result_count(), 3);
    }

    #[test]
    fn context() {
        let context = Context::new();
        AffineMap::empty(&context).context();
    }

    #[test]
    fn dim_count() {
        let context = Context::new();
        let map = AffineMap::zero_result(&context, 4, 0);
        assert_eq!(map.dim_count(), 4);
    }

    #[test]
    fn symbol_count() {
        let context = Context::new();
        let map = AffineMap::zero_result(&context, 0, 3);
        assert_eq!(map.symbol_count(), 3);
    }

    #[test]
    fn result_count() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let map = AffineMap::new(&context, 1, 0, &[d0]);
        assert_eq!(map.result_count(), 1);
    }

    #[test]
    fn input_count() {
        let context = Context::new();
        // 2 dims + 1 symbol = 3 inputs
        let map = AffineMap::zero_result(&context, 2, 1);
        assert_eq!(map.input_count(), 3);
    }

    #[test]
    fn result_access() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let d1 = AffineExpr::dim(&context, 1);
        let map = AffineMap::new(&context, 2, 0, &[d0, d1]);
        assert_eq!(map.result(0).unwrap(), d0);
        assert_eq!(map.result(1).unwrap(), d1);
    }

    #[test]
    fn result_out_of_bounds() {
        let context = Context::new();
        let map = AffineMap::empty(&context);
        assert!(map.result(0).is_err());
    }

    #[test]
    fn single_constant_result() {
        let context = Context::new();
        let map = AffineMap::constant(&context, 7);
        assert_eq!(map.single_constant_result(), 7);
    }

    #[test]
    fn major_sub_map() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let d1 = AffineExpr::dim(&context, 1);
        let map = AffineMap::new(&context, 2, 0, &[d0, d1]);
        let sub = map.major_sub_map(1);
        assert_eq!(sub.result_count(), 1);
        assert_eq!(sub.result(0).unwrap(), d0);
    }

    #[test]
    fn minor_sub_map() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let d1 = AffineExpr::dim(&context, 1);
        let map = AffineMap::new(&context, 2, 0, &[d0, d1]);
        let sub = map.minor_sub_map(1);
        assert_eq!(sub.result_count(), 1);
        assert_eq!(sub.result(0).unwrap(), d1);
    }

    #[test]
    fn sub_map() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let d1 = AffineExpr::dim(&context, 1);
        let d2 = AffineExpr::dim(&context, 2);
        let map = AffineMap::new(&context, 3, 0, &[d0, d1, d2]);
        let sub = map.sub_map(&[0, 2]);
        assert_eq!(sub.result_count(), 2);
        assert_eq!(sub.result(0).unwrap(), d0);
        assert_eq!(sub.result(1).unwrap(), d2);
    }

    #[test]
    fn is_empty() {
        let context = Context::new();
        assert!(AffineMap::empty(&context).is_empty());
        assert!(!AffineMap::multi_dim_identity(&context, 1).is_empty());
    }

    #[test]
    fn is_identity() {
        let context = Context::new();
        assert!(AffineMap::multi_dim_identity(&context, 2).is_identity());
        assert!(!AffineMap::constant(&context, 0).is_identity());
    }

    #[test]
    fn is_minor_identity() {
        let context = Context::new();
        assert!(AffineMap::minor_identity(&context, 3, 2).is_minor_identity());
    }

    #[test]
    fn is_permutation() {
        let context = Context::new();
        assert!(AffineMap::permutation(&context, &[2, 0, 1]).is_permutation());
        assert!(!AffineMap::constant(&context, 0).is_permutation());
    }

    #[test]
    fn is_projected_permutation() {
        let context = Context::new();
        // A permutation is also a projected permutation
        assert!(AffineMap::permutation(&context, &[1, 0]).is_projected_permutation());
    }

    #[test]
    fn is_single_constant() {
        let context = Context::new();
        assert!(AffineMap::constant(&context, 5).is_single_constant());
        assert!(!AffineMap::multi_dim_identity(&context, 1).is_single_constant());
    }

    #[test]
    fn replace() {
        let context = Context::new();
        let d0 = AffineExpr::dim(&context, 0);
        let c1 = AffineExpr::constant(&context, 1);
        // Map: (d0) -> (d0); replace d0 with constant 1 -> () -> (1)
        let map = AffineMap::new(&context, 1, 0, &[d0]);
        let replaced = map.replace(d0, c1, 0, 0);
        assert!(replaced.is_single_constant());
        assert_eq!(replaced.single_constant_result(), 1);
    }

    #[test]
    fn display() {
        let context = Context::new();
        let map = AffineMap::multi_dim_identity(&context, 2);
        assert_eq!(map.to_string(), "(d0, d1) -> (d0, d1)");
    }

    #[test]
    fn debug() {
        let context = Context::new();
        let map = AffineMap::multi_dim_identity(&context, 1);
        assert_eq!(format!("{:?}", map), "(d0) -> (d0)");
    }

    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(
            AffineMap::multi_dim_identity(&context, 2),
            AffineMap::multi_dim_identity(&context, 2),
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(
            AffineMap::multi_dim_identity(&context, 1),
            AffineMap::multi_dim_identity(&context, 2),
        );
    }
}
