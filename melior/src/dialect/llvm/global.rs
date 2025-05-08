use crate::{
    dialect::llvm::attributes::Linkage,
    ir::{Attribute, Region},
};

pub enum GlobalValue<'c> {
    /// A global variable.
    Value(Attribute<'c>),
    /// A global variable with a constant initializer.
    Constant(Attribute<'c>),
    /// A global variable with a constant initializer and a body.
    Complex(Region<'c>),
}

#[derive(Default)]
pub struct GlobalVariableOptions {
    pub(crate) addr_space: Option<i32>,
    pub(crate) alignment: Option<i64>,
    pub(crate) linkage: Option<Linkage>,
}

impl GlobalVariableOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn addr_space(mut self, addr_space: i32) -> Self {
        self.addr_space = Some(addr_space);
        self
    }

    pub fn alignment(mut self, alignment: i64) -> Self {
        self.alignment = Some(alignment);
        self
    }

    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = Some(linkage);
        self
    }
}
