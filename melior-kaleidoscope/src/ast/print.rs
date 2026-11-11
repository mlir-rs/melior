use super::Expression;
use crate::location::Location;

/// A print statement.
#[derive(Debug, PartialEq)]
pub struct Print {
    argument: Expression,
    location: Location,
}

impl Print {
    /// Creates a print statement.
    pub const fn new(argument: Expression, location: Location) -> Self {
        Self { argument, location }
    }

    /// Returns an argument.
    pub const fn argument(&self) -> &Expression {
        &self.argument
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
