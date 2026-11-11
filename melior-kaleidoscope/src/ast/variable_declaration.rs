use super::Type;
use crate::location::Location;

/// A variable declaration.
#[derive(Debug, Eq, PartialEq)]
pub struct VariableDeclaration {
    name: String,
    r#type: Type,
    location: Location,
}

impl VariableDeclaration {
    /// Creates a variable declaration.
    pub const fn new(name: String, r#type: Type, location: Location) -> Self {
        Self {
            name,
            r#type,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a type.
    pub const fn r#type(&self) -> &Type {
        &self.r#type
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
