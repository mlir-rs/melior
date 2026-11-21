use super::{Binary, Call, Literal, Number, StructLiteral, Variable};
use crate::location::Location;

/// An expression.
#[derive(Debug, PartialEq)]
pub enum Expression {
    /// A binary operation.
    Binary(Binary),
    /// A function call.
    Call(Call),
    /// A tensor literal.
    Literal(Literal),
    /// A number.
    Number(Number),
    /// A struct literal.
    StructLiteral(StructLiteral),
    /// A variable reference.
    Variable(Variable),
}

impl Expression {
    /// Returns a location.
    pub const fn location(&self) -> Location {
        match self {
            Self::Binary(binary) => binary.location(),
            Self::Call(call) => call.location(),
            Self::Literal(literal) => literal.location(),
            Self::Number(number) => number.location(),
            Self::StructLiteral(literal) => literal.location(),
            Self::Variable(variable) => variable.location(),
        }
    }
}
