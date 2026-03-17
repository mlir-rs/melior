use crate::{
    context::{Context, ContextRef},
    ir::AffineMap,
    utility::print_callback,
};
use mlir_sys::{
    MlirAffineExpr, mlirAffineAddExprGet, mlirAffineBinaryOpExprGetLHS,
    mlirAffineBinaryOpExprGetRHS, mlirAffineCeilDivExprGet, mlirAffineConstantExprGet,
    mlirAffineConstantExprGetValue, mlirAffineDimExprGet, mlirAffineDimExprGetPosition,
    mlirAffineExprCompose, mlirAffineExprDump, mlirAffineExprEqual, mlirAffineExprGetContext,
    mlirAffineExprGetLargestKnownDivisor, mlirAffineExprIsAAdd, mlirAffineExprIsABinary,
    mlirAffineExprIsACeilDiv, mlirAffineExprIsAConstant, mlirAffineExprIsADim,
    mlirAffineExprIsAFloorDiv, mlirAffineExprIsAMod, mlirAffineExprIsAMul, mlirAffineExprIsASymbol,
    mlirAffineExprIsFunctionOfDim, mlirAffineExprIsMultipleOf, mlirAffineExprIsPureAffine,
    mlirAffineExprIsSymbolicOrConstant, mlirAffineExprPrint, mlirAffineExprShiftDims,
    mlirAffineExprShiftSymbols, mlirAffineFloorDivExprGet, mlirAffineModExprGet,
    mlirAffineMulExprGet, mlirAffineSymbolExprGet, mlirAffineSymbolExprGetPosition,
};
use std::{
    ffi::c_void,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    ops::{Add, Mul},
};

/// An affine expression.
#[derive(Clone, Copy)]
pub struct AffineExpr<'c> {
    raw: MlirAffineExpr,
    _context: PhantomData<&'c Context>,
}

impl<'c> AffineExpr<'c> {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Creates a dimension expression.
    pub fn dim(context: &'c Context, position: usize) -> Self {
        unsafe { Self::from_raw(mlirAffineDimExprGet(context.to_raw(), position as isize)) }
    }

    /// Creates a symbol expression.
    pub fn symbol(context: &'c Context, position: usize) -> Self {
        unsafe { Self::from_raw(mlirAffineSymbolExprGet(context.to_raw(), position as isize)) }
    }

    /// Creates a constant expression.
    pub fn constant(context: &'c Context, value: i64) -> Self {
        unsafe { Self::from_raw(mlirAffineConstantExprGet(context.to_raw(), value)) }
    }

    /// Creates a ceiling division expression.
    pub fn ceil_div(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineCeilDivExprGet(lhs.raw, rhs.raw)) }
    }

    /// Creates a floor division expression.
    pub fn floor_div(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineFloorDivExprGet(lhs.raw, rhs.raw)) }
    }

    /// Creates a modulo expression.
    pub fn modulo(lhs: Self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineModExprGet(lhs.raw, rhs.raw)) }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Returns the context of an affine expression.
    pub fn context(&self) -> ContextRef<'c> {
        unsafe { ContextRef::from_raw(mlirAffineExprGetContext(self.raw)) }
    }

    /// Returns the position of a dimension expression.
    pub fn dim_position(&self) -> usize {
        unsafe { mlirAffineDimExprGetPosition(self.raw) as usize }
    }

    /// Returns the position of a symbol expression.
    pub fn symbol_position(&self) -> usize {
        unsafe { mlirAffineSymbolExprGetPosition(self.raw) as usize }
    }

    /// Returns the value of a constant expression.
    pub fn constant_value(&self) -> i64 {
        unsafe { mlirAffineConstantExprGetValue(self.raw) }
    }

    /// Returns the left-hand side of a binary expression.
    pub fn lhs(&self) -> Self {
        unsafe { Self::from_raw(mlirAffineBinaryOpExprGetLHS(self.raw)) }
    }

    /// Returns the right-hand side of a binary expression.
    pub fn rhs(&self) -> Self {
        unsafe { Self::from_raw(mlirAffineBinaryOpExprGetRHS(self.raw)) }
    }

    /// Returns the largest known divisor of an affine expression.
    pub fn largest_known_divisor(&self) -> i64 {
        unsafe { mlirAffineExprGetLargestKnownDivisor(self.raw) }
    }

    // -----------------------------------------------------------------------
    // Predicates
    // -----------------------------------------------------------------------

    /// Returns `true` if the expression is a dimension expression.
    pub fn is_dim(&self) -> bool {
        unsafe { mlirAffineExprIsADim(self.raw) }
    }

    /// Returns `true` if the expression is a symbol expression.
    pub fn is_symbol(&self) -> bool {
        unsafe { mlirAffineExprIsASymbol(self.raw) }
    }

    /// Returns `true` if the expression is a constant expression.
    pub fn is_constant(&self) -> bool {
        unsafe { mlirAffineExprIsAConstant(self.raw) }
    }

    /// Returns `true` if the expression is an addition expression.
    pub fn is_add(&self) -> bool {
        unsafe { mlirAffineExprIsAAdd(self.raw) }
    }

    /// Returns `true` if the expression is a multiplication expression.
    pub fn is_mul(&self) -> bool {
        unsafe { mlirAffineExprIsAMul(self.raw) }
    }

    /// Returns `true` if the expression is a modulo expression.
    pub fn is_mod(&self) -> bool {
        unsafe { mlirAffineExprIsAMod(self.raw) }
    }

    /// Returns `true` if the expression is a floor division expression.
    pub fn is_floor_div(&self) -> bool {
        unsafe { mlirAffineExprIsAFloorDiv(self.raw) }
    }

    /// Returns `true` if the expression is a ceiling division expression.
    pub fn is_ceil_div(&self) -> bool {
        unsafe { mlirAffineExprIsACeilDiv(self.raw) }
    }

    /// Returns `true` if the expression is a binary expression.
    pub fn is_binary(&self) -> bool {
        unsafe { mlirAffineExprIsABinary(self.raw) }
    }

    /// Returns `true` if the expression is purely affine (no symbols or
    /// modulo).
    pub fn is_pure_affine(&self) -> bool {
        unsafe { mlirAffineExprIsPureAffine(self.raw) }
    }

    /// Returns `true` if the expression is symbolic or constant.
    pub fn is_symbolic_or_constant(&self) -> bool {
        unsafe { mlirAffineExprIsSymbolicOrConstant(self.raw) }
    }

    /// Returns `true` if the expression is a function of the given dimension.
    pub fn is_function_of_dim(&self, position: usize) -> bool {
        unsafe { mlirAffineExprIsFunctionOfDim(self.raw, position as isize) }
    }

    /// Returns `true` if the expression is a multiple of the given factor.
    pub fn is_multiple_of(&self, factor: i64) -> bool {
        unsafe { mlirAffineExprIsMultipleOf(self.raw, factor) }
    }

    // -----------------------------------------------------------------------
    // Transforms
    // -----------------------------------------------------------------------

    /// Composes the affine expression with the given affine map.
    pub fn compose(&self, map: AffineMap<'c>) -> Self {
        unsafe { Self::from_raw(mlirAffineExprCompose(self.raw, map.to_raw())) }
    }

    /// Shifts dimension expressions in the range `[offset, offset + num_dims)`
    /// by `shift`.
    pub fn shift_dims(&self, num_dims: usize, shift: usize, offset: usize) -> Self {
        unsafe {
            Self::from_raw(mlirAffineExprShiftDims(
                self.raw,
                num_dims as u32,
                shift as u32,
                offset as u32,
            ))
        }
    }

    /// Shifts symbol expressions in the range `[offset, offset + num_symbols)`
    /// by `shift`.
    pub fn shift_symbols(&self, num_symbols: usize, shift: usize, offset: usize) -> Self {
        unsafe {
            Self::from_raw(mlirAffineExprShiftSymbols(
                self.raw,
                num_symbols as u32,
                shift as u32,
                offset as u32,
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

    /// Dumps an affine expression to stderr.
    pub fn dump(&self) {
        unsafe { mlirAffineExprDump(self.raw) }
    }

    /// Creates an affine expression from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirAffineExpr) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Converts an affine expression into a raw object.
    pub const fn to_raw(self) -> MlirAffineExpr {
        self.raw
    }
}

impl<'c> Add for AffineExpr<'c> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineAddExprGet(self.raw, rhs.raw)) }
    }
}

impl<'c> Mul for AffineExpr<'c> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        unsafe { Self::from_raw(mlirAffineMulExprGet(self.raw, rhs.raw)) }
    }
}

impl PartialEq for AffineExpr<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirAffineExprEqual(self.raw, other.raw) }
    }
}

impl Eq for AffineExpr<'_> {}

impl Display for AffineExpr<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirAffineExprPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

impl Debug for AffineExpr<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn dim() {
        let context = Context::new();
        let expr = AffineExpr::dim(&context, 0);
        assert!(expr.is_dim());
    }

    #[test]
    fn symbol() {
        let context = Context::new();
        let expr = AffineExpr::symbol(&context, 0);
        assert!(expr.is_symbol());
    }

    #[test]
    fn constant() {
        let context = Context::new();
        let expr = AffineExpr::constant(&context, 42);
        assert!(expr.is_constant());
    }

    #[test]
    fn add() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 1);
        let expr = lhs + rhs;
        assert!(expr.is_add());
        assert!(expr.is_binary());
    }

    #[test]
    fn mul() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 2);
        let expr = lhs * rhs;
        assert!(expr.is_mul());
        assert!(expr.is_binary());
    }

    #[test]
    fn ceil_div() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 4);
        let expr = AffineExpr::ceil_div(lhs, rhs);
        assert!(expr.is_ceil_div());
        assert!(expr.is_binary());
    }

    #[test]
    fn floor_div() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 4);
        let expr = AffineExpr::floor_div(lhs, rhs);
        assert!(expr.is_floor_div());
        assert!(expr.is_binary());
    }

    #[test]
    fn modulo() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 4);
        let expr = AffineExpr::modulo(lhs, rhs);
        assert!(expr.is_mod());
        assert!(expr.is_binary());
    }

    #[test]
    fn context() {
        let context = Context::new();
        AffineExpr::dim(&context, 0).context();
    }

    #[test]
    fn dim_position() {
        let context = Context::new();
        let expr = AffineExpr::dim(&context, 3);
        assert_eq!(expr.dim_position(), 3);
    }

    #[test]
    fn symbol_position() {
        let context = Context::new();
        let expr = AffineExpr::symbol(&context, 2);
        assert_eq!(expr.symbol_position(), 2);
    }

    #[test]
    fn constant_value() {
        let context = Context::new();
        let expr = AffineExpr::constant(&context, 42);
        assert_eq!(expr.constant_value(), 42);
    }

    #[test]
    fn lhs_rhs() {
        let context = Context::new();
        let lhs = AffineExpr::dim(&context, 0);
        let rhs = AffineExpr::constant(&context, 1);
        let expr = lhs + rhs;
        assert_eq!(expr.lhs(), lhs);
        assert_eq!(expr.rhs(), rhs);
    }

    #[test]
    fn largest_known_divisor() {
        let context = Context::new();
        // constant(6) has divisor 6
        let expr = AffineExpr::constant(&context, 6);
        assert_eq!(expr.largest_known_divisor(), 6);
    }

    #[test]
    fn is_pure_affine() {
        let context = Context::new();
        // dim + constant is pure affine
        let expr = AffineExpr::dim(&context, 0) + AffineExpr::constant(&context, 1);
        assert!(expr.is_pure_affine());
    }

    #[test]
    fn is_symbolic_or_constant() {
        let context = Context::new();
        let expr = AffineExpr::constant(&context, 5);
        assert!(expr.is_symbolic_or_constant());
        let sym = AffineExpr::symbol(&context, 0);
        assert!(sym.is_symbolic_or_constant());
        // dim is not symbolic or constant
        let dim = AffineExpr::dim(&context, 0);
        assert!(!dim.is_symbolic_or_constant());
    }

    #[test]
    fn is_function_of_dim() {
        let context = Context::new();
        let expr = AffineExpr::dim(&context, 0);
        assert!(expr.is_function_of_dim(0));
        assert!(!expr.is_function_of_dim(1));
    }

    #[test]
    fn is_multiple_of() {
        let context = Context::new();
        // 2 * d0 is a multiple of 2
        let expr = AffineExpr::constant(&context, 2) * AffineExpr::dim(&context, 0);
        assert!(expr.is_multiple_of(2));
    }

    #[test]
    fn equal() {
        let context = Context::new();
        assert_eq!(AffineExpr::dim(&context, 0), AffineExpr::dim(&context, 0),);
        assert_eq!(
            AffineExpr::constant(&context, 7),
            AffineExpr::constant(&context, 7),
        );
    }

    #[test]
    fn not_equal() {
        let context = Context::new();
        assert_ne!(AffineExpr::dim(&context, 0), AffineExpr::dim(&context, 1),);
        assert_ne!(
            AffineExpr::dim(&context, 0),
            AffineExpr::constant(&context, 0),
        );
    }

    #[test]
    fn display() {
        let context = Context::new();
        assert_eq!(AffineExpr::dim(&context, 0).to_string(), "d0");
        assert_eq!(AffineExpr::symbol(&context, 0).to_string(), "s0");
        assert_eq!(AffineExpr::constant(&context, 42).to_string(), "42");
    }

    #[test]
    fn debug() {
        let context = Context::new();
        let expr = AffineExpr::dim(&context, 0);
        assert_eq!(format!("{:?}", expr), "d0");
    }

    #[test]
    fn shift_dims() {
        let context = Context::new();
        // d0 shifted by 1 in range [0, 1) becomes d1
        let expr = AffineExpr::dim(&context, 0);
        let shifted = expr.shift_dims(1, 1, 0);
        assert_eq!(shifted, AffineExpr::dim(&context, 1));
    }

    #[test]
    fn shift_symbols() {
        let context = Context::new();
        // s0 shifted by 1 in range [0, 1) becomes s1
        let expr = AffineExpr::symbol(&context, 0);
        let shifted = expr.shift_symbols(1, 1, 0);
        assert_eq!(shifted, AffineExpr::symbol(&context, 1));
    }

    #[test]
    fn compose() {
        use mlir_sys::mlirAffineMapMultiDimIdentityGet;
        let context = Context::new();
        // d0 composed with the 1-dim identity map (d0 -> d0) gives d0
        let expr = AffineExpr::dim(&context, 0);
        let map =
            unsafe { AffineMap::from_raw(mlirAffineMapMultiDimIdentityGet(context.to_raw(), 1)) };
        let composed = expr.compose(map);
        assert_eq!(composed, expr);
    }
}
