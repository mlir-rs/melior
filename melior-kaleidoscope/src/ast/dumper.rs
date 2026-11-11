use super::{
    Expression, Function, Module, Prototype, Record, Statement, Struct, VariableDeclaration,
    VariableType,
};
use crate::location::Location;
use core::fmt::{self, Formatter};

// The dump format follows the one of the AST dumper in the MLIR Toy tutorial.
pub(super) struct Dumper<'a, 'b> {
    formatter: &'a mut Formatter<'b>,
    file_name: &'a str,
    level: usize,
}

impl<'a, 'b> Dumper<'a, 'b> {
    pub(super) const fn new(formatter: &'a mut Formatter<'b>, file_name: &'a str) -> Self {
        Self {
            formatter,
            file_name,
            level: 0,
        }
    }

    pub(super) fn dump_module(&mut self, module: &Module) -> fmt::Result {
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
    use crate::parse::parse;
    use pretty_assertions::assert_eq;

    #[test]
    fn dump_example() {
        insta::assert_snapshot!(
            parse(include_str!("../../examples/ast.toy"), "ast.toy")
                .unwrap()
                .to_string()
        );
    }

    #[test]
    fn dump_struct_example() {
        insta::assert_snapshot!(
            parse(
                include_str!("../../examples/struct_codegen.toy"),
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
