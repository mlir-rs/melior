use crate::{Context, ir::Attribute};

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
pub fn linkage(context: &Context, linkage: Linkage) -> Attribute<'_> {
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
