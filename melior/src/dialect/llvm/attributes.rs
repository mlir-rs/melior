use crate::{ir::Attribute, Context};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Private,
    Internal,
    AvailableExternally,
    LinkOnce,
    Weak,
    Common,
    Appending,
    External,
}

/// Creates an LLVM linkage attribute.
pub fn linkage(context: &Context, linkage: Linkage) -> Attribute {
    let linkage = match linkage {
        Linkage::Private => "private",
        Linkage::Internal => "internal",
        Linkage::AvailableExternally => "available_externally",
        Linkage::LinkOnce => "link_once",
        Linkage::Weak => "weak",
        Linkage::Common => "common",
        Linkage::Appending => "appending",
        Linkage::External => "external",
    };
    Attribute::parse(context, &format!("#llvm.linkage<{linkage}>")).unwrap()
}

// https://github.com/llvm/llvm-project/blob/600eeed51f538adc5f43c8223a57608e73aba31f/mlir/include/mlir/Dialect/LLVMIR/LLVMEnums.td#L542-L558
/// LLVM float comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum FCmpPredicate {
    /// Always returns false.
    False = 0,
    /// Ordered and equal.
    Oeq = 1,
    /// Ordered and greater than.
    Ogt = 2,
    /// Ordered and greater than or equal.
    Oge = 3,
    /// Ordered and less than.
    Olt = 4,
    /// Ordered and less than or equal.
    Ole = 5,
    /// Ordered and not equal.
    One = 6,
    /// Ordered (no NaNs).
    Ord = 7,
    /// Unordered or equal.
    Ueq = 8,
    /// Unordered or greater than.
    Ugt = 9,
    /// Unordered or greater than or equal.
    Uge = 10,
    /// Unordered or less than.
    Ult = 11,
    /// Unordered or less than or equal.
    Ule = 12,
    /// Unordered or not equal.
    Une = 13,
    /// Unordered (either NaNs).
    Uno = 14,
    /// Always returns true.
    True = 15,
}
/// LLVM integer comparison predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum ICmpPredicate {
    /// Equal
    Eq = 0,
    /// Not equal
    Ne = 1,
    /// Signed less than
    Slt = 2,
    /// Signed less than or equal
    Sle = 3,
    /// Signed greater than
    Sgt = 4,
    /// Signed greater than or equal
    Sge = 5,
    /// Unsigned less than
    Ult = 6,
    /// Unsigned less than or equal
    Ule = 7,
    /// Unsigned greater than
    Ugt = 8,
    /// Unsigned greater than or equal
    Uge = 9,
}
