use crate::location::Location;

/// A variable reference.
#[derive(Debug, Eq, PartialEq)]
pub struct Variable {
    name: String,
    location: Location,
}

impl Variable {
    /// Creates a variable reference.
    pub const fn new(name: String, location: Location) -> Self {
        Self { name, location }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
