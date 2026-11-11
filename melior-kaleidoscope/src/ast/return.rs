use super::Expression;
use crate::location::Location;

/// A return statement.
#[derive(Debug, PartialEq)]
pub struct Return {
    value: Option<Expression>,
    location: Location,
}

impl Return {
    /// Creates a return statement.
    pub const fn new(value: Option<Expression>, location: Location) -> Self {
        Self { value, location }
    }

    /// Returns a value.
    pub const fn value(&self) -> Option<&Expression> {
        self.value.as_ref()
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
