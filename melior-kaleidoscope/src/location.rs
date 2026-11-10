use core::fmt::{self, Display, Formatter};

/// A location in a source file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Location {
    line: usize,
    column: usize,
}

impl Location {
    /// Creates a location.
    pub const fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    /// Returns a line number.
    pub const fn line(&self) -> usize {
        self.line
    }

    /// Returns a column number.
    pub const fn column(&self) -> usize {
        self.column
    }
}

impl Display for Location {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "{}:{}", self.line, self.column)
    }
}
