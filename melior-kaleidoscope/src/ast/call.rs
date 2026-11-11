use super::Expression;
use crate::location::Location;

/// A function call.
#[derive(Debug, PartialEq)]
pub struct Call {
    callee: String,
    arguments: Vec<Expression>,
    location: Location,
}

impl Call {
    /// Creates a function call.
    pub const fn new(callee: String, arguments: Vec<Expression>, location: Location) -> Self {
        Self {
            callee,
            arguments,
            location,
        }
    }

    /// Returns a callee name.
    pub fn callee(&self) -> &str {
        &self.callee
    }

    /// Returns arguments.
    pub fn arguments(&self) -> &[Expression] {
        &self.arguments
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
