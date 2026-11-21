use super::{Expression, Print, Return, VariableDefinition};

/// A statement in a function body.
#[derive(Debug, PartialEq)]
pub enum Statement {
    /// An expression.
    Expression(Expression),
    /// A print statement.
    Print(Print),
    /// A return statement.
    Return(Return),
    /// A variable definition.
    VariableDefinition(VariableDefinition),
}
