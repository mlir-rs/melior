//! An abstract syntax tree of the Toy language.

use crate::location::Location;
use core::fmt::{self, Display, Formatter};

/// A module.
#[derive(Debug, PartialEq)]
pub struct Module {
    file_name: String,
    records: Vec<Record>,
}

impl Module {
    /// Creates a module.
    pub const fn new(file_name: String, records: Vec<Record>) -> Self {
        Self { file_name, records }
    }

    /// Returns a file name.
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Returns records.
    pub fn records(&self) -> &[Record] {
        &self.records
    }
}

impl Display for Module {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Dumper::new(formatter, &self.file_name).dump_module(self)
    }
}

/// A top-level record.
#[derive(Debug, PartialEq)]
pub enum Record {
    /// A function definition.
    Function(Function),
    /// A struct definition.
    Struct(Struct),
}

/// A function definition.
#[derive(Debug, PartialEq)]
pub struct Function {
    prototype: Prototype,
    body: Vec<Statement>,
}

impl Function {
    /// Creates a function.
    pub const fn new(prototype: Prototype, body: Vec<Statement>) -> Self {
        Self { prototype, body }
    }

    /// Returns a prototype.
    pub const fn prototype(&self) -> &Prototype {
        &self.prototype
    }

    /// Returns a body.
    pub fn body(&self) -> &[Statement] {
        &self.body
    }
}

/// A function prototype.
#[derive(Debug, PartialEq)]
pub struct Prototype {
    name: String,
    parameters: Vec<VariableDeclaration>,
    location: Location,
}

impl Prototype {
    /// Creates a prototype.
    pub const fn new(
        name: String,
        parameters: Vec<VariableDeclaration>,
        location: Location,
    ) -> Self {
        Self {
            name,
            parameters,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns parameters.
    pub fn parameters(&self) -> &[VariableDeclaration] {
        &self.parameters
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

/// A struct definition.
#[derive(Debug, PartialEq)]
pub struct Struct {
    name: String,
    variables: Vec<VariableDeclaration>,
    location: Location,
}

impl Struct {
    /// Creates a struct.
    pub const fn new(
        name: String,
        variables: Vec<VariableDeclaration>,
        location: Location,
    ) -> Self {
        Self {
            name,
            variables,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns variables.
    pub fn variables(&self) -> &[VariableDeclaration] {
        &self.variables
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

/// A variable type.
#[derive(Debug, Eq, PartialEq)]
pub enum VariableType {
    /// A struct type of a name.
    Struct(String),
    /// A tensor type of a shape, which is unknown if empty.
    Tensor(Vec<u64>),
}

/// A variable declaration.
#[derive(Debug, PartialEq)]
pub struct VariableDeclaration {
    name: String,
    r#type: VariableType,
    initializer: Option<Expression>,
    location: Location,
}

impl VariableDeclaration {
    /// Creates a variable declaration.
    pub const fn new(
        name: String,
        r#type: VariableType,
        initializer: Option<Expression>,
        location: Location,
    ) -> Self {
        Self {
            name,
            r#type,
            initializer,
            location,
        }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a type.
    pub const fn r#type(&self) -> &VariableType {
        &self.r#type
    }

    /// Returns an initializer.
    pub const fn initializer(&self) -> Option<&Expression> {
        self.initializer.as_ref()
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

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

/// A print statement.
#[derive(Debug, PartialEq)]
pub struct Print {
    argument: Expression,
    location: Location,
}

impl Print {
    /// Creates a print statement.
    pub const fn new(argument: Expression, location: Location) -> Self {
        Self { argument, location }
    }

    /// Returns an argument.
    pub const fn argument(&self) -> &Expression {
        &self.argument
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

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

/// A binary operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOperator {
    /// A struct member access.
    Access,
    /// An addition.
    Add,
    /// A multiplication.
    Multiply,
    /// A subtraction.
    Subtract,
}

impl Display for BinaryOperator {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(
            formatter,
            "{}",
            match self {
                Self::Access => '.',
                Self::Add => '+',
                Self::Multiply => '*',
                Self::Subtract => '-',
            }
        )
    }
}

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

/// A struct literal.
#[derive(Debug, PartialEq)]
pub struct StructLiteral {
    values: Vec<Expression>,
    location: Location,
}

impl StructLiteral {
    /// Creates a struct literal.
    pub const fn new(values: Vec<Expression>, location: Location) -> Self {
        Self { values, location }
    }

    /// Returns values.
    pub fn values(&self) -> &[Expression] {
        &self.values
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

/// A variable reference.
#[derive(Debug, Eq, PartialEq)]
pub struct Variable {
    name: String,
    location: Location,
}

impl Variable {
    /// Creates a variable reference.
    pub const fn new(name: String, location: Location) -> Self {
        Self { name, location }
    }

    /// Returns a name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a location.
    pub const fn location(&self) -> Location {
        self.location
    }
}

// The dump format follows the one of the AST dumper in the MLIR Toy tutorial.
struct Dumper<'a, 'b> {
    formatter: &'a mut Formatter<'b>,
    file_name: &'a str,
    level: usize,
}

impl<'a, 'b> Dumper<'a, 'b> {
    const fn new(formatter: &'a mut Formatter<'b>, file_name: &'a str) -> Self {
        Self {
            formatter,
            file_name,
            level: 0,
        }
    }

    fn dump_module(&mut self, module: &Module) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            writeln!(dumper.formatter, "Module:")?;

            module.records().iter().try_for_each(|record| match record {
                Record::Function(function) => dumper.dump_function(function),
                Record::Struct(r#struct) => dumper.dump_struct(r#struct),
            })
        })
    }

    fn dump_function(&mut self, function: &Function) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            writeln!(dumper.formatter, "Function ")?;
            dumper.dump_prototype(function.prototype())?;
            dumper.dump_block(function.body())
        })
    }

    fn dump_prototype(&mut self, prototype: &Prototype) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            writeln!(
                dumper.formatter,
                "Proto '{}' {}",
                prototype.name(),
                dumper.location(prototype.location())
            )?;
            dumper.indent()?;
            writeln!(
                dumper.formatter,
                "Params: [{}]",
                prototype
                    .parameters()
                    .iter()
                    .map(VariableDeclaration::name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }

    fn dump_struct(&mut self, r#struct: &Struct) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            writeln!(
                dumper.formatter,
                "Struct: {} {}",
                r#struct.name(),
                dumper.location(r#struct.location())
            )?;

            dumper.nest(|dumper| {
                dumper.indent()?;
                writeln!(dumper.formatter, "Variables: [")?;
                r#struct
                    .variables()
                    .iter()
                    .try_for_each(|variable| dumper.dump_variable_declaration(variable))?;
                dumper.indent()?;
                writeln!(dumper.formatter, "]")
            })
        })
    }

    fn dump_block(&mut self, statements: &[Statement]) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            writeln!(dumper.formatter, "Block {{")?;
            statements
                .iter()
                .try_for_each(|statement| dumper.dump_statement(statement))?;
            dumper.indent()?;
            writeln!(dumper.formatter, "}} // Block")
        })
    }

    fn dump_statement(&mut self, statement: &Statement) -> fmt::Result {
        match statement {
            Statement::Expression(expression) => self.dump_expression(expression),
            Statement::Print(print) => self.nest(|dumper| {
                dumper.indent()?;
                writeln!(
                    dumper.formatter,
                    "Print [ {}",
                    dumper.location(print.location())
                )?;
                dumper.dump_expression(print.argument())?;
                dumper.indent()?;
                writeln!(dumper.formatter, "]")
            }),
            Statement::Return(r#return) => self.nest(|dumper| {
                dumper.indent()?;
                writeln!(dumper.formatter, "Return")?;

                if let Some(value) = r#return.value() {
                    dumper.dump_expression(value)
                } else {
                    dumper.nest(|dumper| {
                        dumper.indent()?;
                        writeln!(dumper.formatter, "(void)")
                    })
                }
            }),
            Statement::VariableDeclaration(declaration) => {
                self.dump_variable_declaration(declaration)
            }
        }
    }

    fn dump_variable_declaration(&mut self, declaration: &VariableDeclaration) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;
            write!(dumper.formatter, "VarDecl {}<", declaration.name())?;

            match declaration.r#type() {
                VariableType::Struct(name) => write!(dumper.formatter, "{name}")?,
                VariableType::Tensor(shape) => write!(
                    dumper.formatter,
                    "{}",
                    shape
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )?,
            }

            writeln!(
                dumper.formatter,
                "> {}",
                dumper.location(declaration.location())
            )?;

            if let Some(initializer) = declaration.initializer() {
                dumper.dump_expression(initializer)?;
            }

            Ok(())
        })
    }

    fn dump_expression(&mut self, expression: &Expression) -> fmt::Result {
        self.nest(|dumper| {
            dumper.indent()?;

            match expression {
                Expression::Binary(binary) => {
                    writeln!(
                        dumper.formatter,
                        "BinOp: {} {}",
                        binary.operator(),
                        dumper.location(binary.location())
                    )?;
                    dumper.dump_expression(binary.lhs())?;
                    dumper.dump_expression(binary.rhs())
                }
                Expression::Call(call) => {
                    writeln!(
                        dumper.formatter,
                        "Call '{}' [ {}",
                        call.callee(),
                        dumper.location(call.location())
                    )?;
                    call.arguments()
                        .iter()
                        .try_for_each(|argument| dumper.dump_expression(argument))?;
                    dumper.indent()?;
                    writeln!(dumper.formatter, "]")
                }
                Expression::Literal(literal) => {
                    write!(dumper.formatter, "Literal: ")?;
                    dumper.dump_literal(expression)?;
                    writeln!(dumper.formatter, " {}", dumper.location(literal.location()))
                }
                Expression::Number(number) => writeln!(
                    dumper.formatter,
                    "{} {}",
                    format_number(number.value()),
                    dumper.location(number.location())
                ),
                Expression::StructLiteral(literal) => {
                    write!(dumper.formatter, "Struct Literal: ")?;
                    literal
                        .values()
                        .iter()
                        .try_for_each(|value| dumper.dump_expression(value))?;
                    dumper.indent()?;
                    writeln!(dumper.formatter, " {}", dumper.location(literal.location()))
                }
                Expression::Variable(variable) => writeln!(
                    dumper.formatter,
                    "var: {} {}",
                    variable.name(),
                    dumper.location(variable.location())
                ),
            }
        })
    }

    fn dump_literal(&mut self, expression: &Expression) -> fmt::Result {
        match expression {
            Expression::Number(number) => {
                write!(self.formatter, "{}", format_number(number.value()))
            }
            Expression::Literal(literal) => {
                write!(
                    self.formatter,
                    "<{}>[ ",
                    literal
                        .dimensions()
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )?;

                for (index, value) in literal.values().iter().enumerate() {
                    if index > 0 {
                        write!(self.formatter, ", ")?;
                    }

                    self.dump_literal(value)?;
                }

                write!(self.formatter, " ]")
            }
            _ => write!(self.formatter, "<invalid literal element>"),
        }
    }

    fn nest(&mut self, callback: impl FnOnce(&mut Self) -> fmt::Result) -> fmt::Result {
        self.level += 1;
        let result = callback(self);
        self.level -= 1;
        result
    }

    fn indent(&mut self) -> fmt::Result {
        (0..self.level).try_for_each(|_| self.formatter.write_str("  "))
    }

    fn location(&self, location: Location) -> String {
        format!("@{}:{location}", self.file_name)
    }
}

// Formats a number like the `%e` conversion in C, which the tutorial uses.
fn format_number(number: f64) -> String {
    let formatted = format!("{number:.6e}");
    let (mantissa, exponent) = formatted
        .split_once('e')
        .expect("exponent in scientific notation");
    let exponent = exponent
        .parse::<i32>()
        .expect("integer exponent in scientific notation");

    format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { '-' } else { '+' },
        exponent.abs()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use pretty_assertions::assert_eq;

    #[test]
    fn dump_example() {
        insta::assert_snapshot!(
            parse(include_str!("../examples/ast.toy"), "ast.toy")
                .unwrap()
                .to_string()
        );
    }

    #[test]
    fn dump_struct_example() {
        insta::assert_snapshot!(
            parse(
                include_str!("../examples/struct_codegen.toy"),
                "struct_codegen.toy"
            )
            .unwrap()
            .to_string()
        );
    }

    #[test]
    fn dump_void_return() {
        insta::assert_snapshot!(
            parse("def main() { return; }", "test.toy")
                .unwrap()
                .to_string()
        );
    }

    #[test]
    fn format_numbers() {
        assert_eq!(format_number(0.0), "0.000000e+00");
        assert_eq!(format_number(1.0), "1.000000e+00");
        assert_eq!(format_number(-1.5), "-1.500000e+00");
        assert_eq!(format_number(123.456), "1.234560e+02");
        assert_eq!(format_number(0.001), "1.000000e-03");
    }
}
