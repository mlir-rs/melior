use super::{BinaryOperator, Expression};
use crate::location::Location;

/// A binary operation.
#[derive(Debug, PartialEq)]
pub struct Binary {
    operator: BinaryOperator,
    lhs: Box<Expression>,
    rhs: Box<Expression>,
    location: Location,
}

impl Binary {
    /// Creates a binary operation.
    pub fn new(
        operator: BinaryOperator,
        lhs: Expression,
        rhs: Expression,
        location: Location,
    ) -> Self {
        Self {
            operator,
            lhs: lhs.into(),
            rhs: rhs.into(),
            location,
        }
    }

    /// Returns an operator.
    pub const fn operator(&self) -> BinaryOperator {
        self.operator
    }

    /// Returns a left-hand side.
    pub const fn lhs(&self) -> &Expression {
        &self.lhs
    }

    /// Returns a right-hand side.
    pub const fn rhs(&self) -> &Expression {
        &self.rhs
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}
