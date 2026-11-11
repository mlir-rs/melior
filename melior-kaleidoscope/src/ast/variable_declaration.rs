use super::{Expression, VariableType};
use crate::location::Location;

/// A variable declaration.
#[derive(Debug, PartialEq)]
pub struct VariableDeclaration {
    name: String,
    r#type: VariableType,
    initializer: Option<Expression>,
    location: Location,
}

impl VariableDeclaration {
    /// Creates a variable declaration.
    pub const fn new(
        name: String,
        r#type: VariableType,
        initializer: Option<Expression>,
        location: Location,
    ) -> Self {
        Self {
            name,
            r#type,
            initializer,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a type.
    pub const fn r#type(&self) -> &VariableType {
        &self.r#type
    }

    /// Returns an initializer.
    pub const fn initializer(&self) -> Option<&Expression> {
        self.initializer.as_ref()
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
