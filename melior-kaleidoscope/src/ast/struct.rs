use super::VariableDeclaration;
use crate::location::Location;

/// A struct definition.
#[derive(Debug, Eq, PartialEq)]
pub struct Struct {
    name: String,
    variables: Vec<VariableDeclaration>,
    location: Location,
}

impl Struct {
    /// Creates a struct.
    pub const fn new(
        name: String,
        variables: Vec<VariableDeclaration>,
        location: Location,
    ) -> Self {
        Self {
            name,
            variables,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns variables.
    pub fn variables(&self) -> &[VariableDeclaration] {
        &self.variables
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
