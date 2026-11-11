use super::{Expression, VariableDeclaration};

/// A variable definition.
#[derive(Debug, PartialEq)]
pub struct VariableDefinition {
    declaration: VariableDeclaration,
    initializer: Expression,
}

impl VariableDefinition {
    /// Creates a variable definition.
    pub const fn new(declaration: VariableDeclaration, initializer: Expression) -> Self {
        Self {
            declaration,
            initializer,
        }
    }

    /// Returns a declaration.
    pub const fn declaration(&self) -> &VariableDeclaration {
        &self.declaration
    }

    /// Returns an initializer.
    pub const fn initializer(&self) -> &Expression {
        &self.initializer
    }
}
