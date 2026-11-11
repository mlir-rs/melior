use super::{Expression, Print, Return, VariableDeclaration};

/// A statement in a function body.
#[derive(Debug, PartialEq)]
pub enum Statement {
    /// An expression.
    Expression(Expression),
    /// A print statement.
    Print(Print),
    /// A return statement.
    Return(Return),
    /// A variable declaration.
    VariableDeclaration(VariableDeclaration),
}
