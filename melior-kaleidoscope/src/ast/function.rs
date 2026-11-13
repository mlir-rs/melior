use super::{Prototype, Statement};

/// A function definition.
#[derive(Debug, PartialEq)]
pub struct Function {
    prototype: Prototype,
    body: Vec<Statement>,
}

impl Function {
    /// Creates a function.
    pub const fn new(prototype: Prototype, body: Vec<Statement>) -> Self {
        Self { prototype, body }
    }

    /// Returns a prototype.
    pub const fn prototype(&self) -> &Prototype {
        &self.prototype
    }

    /// Returns a body.
    pub fn body(&self) -> &[Statement] {
        &self.body
    }
}
