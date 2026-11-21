use crate::location::Location;

/// A number.
#[derive(Debug, PartialEq)]
pub struct Number {
    value: f64,
    location: Location,
}

impl Number {
    /// Creates a number.
    pub const fn new(value: f64, location: Location) -> Self {
        Self { value, location }
    }

    /// Returns a value.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
