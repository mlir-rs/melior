use super::Expression;
use crate::location::Location;

/// A tensor literal.
#[derive(Debug, PartialEq)]
pub struct Literal {
    values: Vec<Expression>,
    dimensions: Vec<u64>,
    location: Location,
}

impl Literal {
    /// Creates a tensor literal.
    pub const fn new(values: Vec<Expression>, dimensions: Vec<u64>, location: Location) -> Self {
        Self {
            values,
            dimensions,
            location,
        }
    }

    /// Returns values, which are numbers or nested literals.
    pub fn values(&self) -> &[Expression] {
        &self.values
    }

    /// Returns dimensions.
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
