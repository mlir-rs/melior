use super::Expression;
use crate::location::Location;

/// A struct literal.
#[derive(Debug, PartialEq)]
pub struct StructLiteral {
    values: Vec<Expression>,
    location: Location,
}

impl StructLiteral {
    /// Creates a struct literal.
    pub const fn new(values: Vec<Expression>, location: Location) -> Self {
        Self { values, location }
    }

    /// Returns values.
    pub fn values(&self) -> &[Expression] {
        &self.values
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
