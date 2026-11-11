//! An AST.

mod binary;
mod binary_operator;
mod call;
mod expression;
mod function;
mod literal;
mod module;
mod number;
mod print;
mod prototype;
mod record;
mod r#return;
mod statement;
mod r#struct;
mod struct_literal;
mod r#type;
mod variable;
mod variable_declaration;

pub use self::{
    binary::Binary, binary_operator::BinaryOperator, call::Call, expression::Expression,
    function::Function, literal::Literal, module::Module, number::Number, print::Print,
    prototype::Prototype, record::Record, r#return::Return, statement::Statement, r#struct::Struct,
    struct_literal::StructLiteral, r#type::Type, variable::Variable,
    variable_declaration::VariableDeclaration,
};
