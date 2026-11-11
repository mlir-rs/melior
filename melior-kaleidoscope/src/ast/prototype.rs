use super::VariableDeclaration;
use crate::location::Location;

/// A function prototype.
#[derive(Debug, Eq, PartialEq)]
pub struct Prototype {
    name: String,
    parameters: Vec<VariableDeclaration>,
    location: Location,
}

impl Prototype {
    /// Creates a prototype.
    pub const fn new(
        name: String,
        parameters: Vec<VariableDeclaration>,
        location: Location,
    ) -> Self {
        Self {
            name,
            parameters,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns parameters.
    pub fn parameters(&self) -> &[VariableDeclaration] {
        &self.parameters
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
