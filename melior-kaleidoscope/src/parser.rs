//! A recursive descent parser of the Toy language.

use crate::{
    ast::{
        Binary, BinaryOperator, Call, Expression, Function, Literal, Module, Number, Print,
        Prototype, Record, Return, Statement, Struct, StructLiteral, Variable, VariableDeclaration,
        VariableType,
    },
    error::Error,
    lexer::{Lexer, Token},
    location::Location,
};

/// Parses a module.
pub fn parse(source: &str, file_name: &str) -> Result<Module, Error> {
    Parser::new(source)?.parse_module(file_name)
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    token: Token,
    location: Location,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Result<Self, Error> {
        let mut lexer = Lexer::new(source);
        let (token, location) = lexer.next_token()?;

        Ok(Self {
            lexer,
            token,
            location,
        })
    }

    fn parse_module(&mut self, file_name: &str) -> Result<Module, Error> {
        let mut records = vec![];

        loop {
            records.push(match self.token {
                Token::End => break,
                Token::Def => Record::Function(self.parse_definition()?),
                Token::Struct => Record::Struct(self.parse_struct()?),
                _ => {
                    return self.fail("'def' or 'struct'", "when parsing top level module records");
                }
            });
        }

        Ok(Module::new(file_name.into(), records))
    }

    fn parse_definition(&mut self) -> Result<Function, Error> {
        Ok(Function::new(self.parse_prototype()?, self.parse_block()?))
    }

    fn parse_prototype(&mut self) -> Result<Prototype, Error> {
        let location = self.location;

        self.expect(Token::Def, "in prototype")?;

        let name = self.parse_identifier("function name", "in prototype")?;

        self.expect(Token::ParenthesisOpen, "in prototype")?;

        let mut parameters = vec![];

        if self.token != Token::ParenthesisClose {
            loop {
                let location = self.location;
                let name_or_type = self.parse_identifier("identifier", "in parameter list")?;

                parameters.push(if matches!(self.token, Token::Identifier(_)) {
                    VariableDeclaration::new(
                        self.parse_identifier("name", "in parameter list")?,
                        VariableType::Struct(name_or_type),
                        None,
                        location,
                    )
                } else {
                    VariableDeclaration::new(
                        name_or_type,
                        VariableType::Tensor(vec![]),
                        None,
                        location,
                    )
                });

                if self.token != Token::Comma {
                    break;
                }

                self.advance()?;

                if !matches!(self.token, Token::Identifier(_)) {
                    return self.fail("identifier", "after ',' in function parameter list");
                }
            }
        }

        self.expect(Token::ParenthesisClose, "to end function prototype")?;

        Ok(Prototype::new(name, parameters, location))
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, Error> {
        self.expect(Token::BraceOpen, "to begin block")?;

        let mut statements = vec![];

        self.skip_semicolons()?;

        while !matches!(self.token, Token::BraceClose | Token::End) {
            statements.push(match self.token {
                Token::Identifier(_) => self.parse_declaration_or_call()?,
                Token::Var => Statement::VariableDeclaration(self.parse_var_declaration(true)?),
                Token::Return => Statement::Return(self.parse_return()?),
                _ => Statement::Expression(self.parse_expression()?),
            });

            if self.token != Token::Semicolon {
                return self.fail("';'", "after expression");
            }

            self.skip_semicolons()?;
        }

        self.expect(Token::BraceClose, "to close block")?;

        Ok(statements)
    }

    fn skip_semicolons(&mut self) -> Result<(), Error> {
        while self.token == Token::Semicolon {
            self.advance()?;
        }

        Ok(())
    }

    fn parse_return(&mut self) -> Result<Return, Error> {
        let location = self.location;

        self.expect(Token::Return, "in return statement")?;

        Ok(Return::new(
            if self.token == Token::Semicolon {
                None
            } else {
                Some(self.parse_expression()?)
            },
            location,
        ))
    }

    fn parse_declaration_or_call(&mut self) -> Result<Statement, Error> {
        let location = self.location;
        let name = self.parse_identifier("identifier", "in statement")?;

        if self.token != Token::ParenthesisOpen {
            return Ok(Statement::VariableDeclaration(
                self.parse_typed_declaration(name, true, location)?,
            ));
        }

        let mut arguments = self.parse_arguments()?;

        if name == "print" {
            if arguments.len() != 1 {
                return self.fail("<single arg>", "as argument to print()");
            }

            Ok(Statement::Print(Print::new(arguments.remove(0), location)))
        } else {
            Ok(Statement::Expression(Expression::Call(Call::new(
                name, arguments, location,
            ))))
        }
    }

    // Parses a variable declaration with a struct type.
    fn parse_typed_declaration(
        &mut self,
        type_name: String,
        requires_initializer: bool,
        location: Location,
    ) -> Result<VariableDeclaration, Error> {
        let name = self.parse_identifier("name", "in variable declaration")?;

        let initializer = if requires_initializer {
            if self.token != Token::Equal {
                return self.fail("initializer", "in variable declaration");
            }

            self.advance()?;

            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(VariableDeclaration::new(
            name,
            VariableType::Struct(type_name),
            initializer,
            location,
        ))
    }

    fn parse_declaration(
        &mut self,
        requires_initializer: bool,
    ) -> Result<VariableDeclaration, Error> {
        if self.token == Token::Var {
            return self.parse_var_declaration(requires_initializer);
        }

        let location = self.location;
        let type_name = self.parse_identifier("type name", "in variable declaration")?;

        self.parse_typed_declaration(type_name, requires_initializer, location)
    }

    fn parse_var_declaration(
        &mut self,
        requires_initializer: bool,
    ) -> Result<VariableDeclaration, Error> {
        let location = self.location;

        self.expect(Token::Var, "to begin declaration")?;

        let name = self.parse_identifier("identifier", "after 'var' declaration")?;

        let shape = if self.token == Token::AngleBracketOpen {
            self.parse_type()?
        } else {
            vec![]
        };

        let initializer = if requires_initializer {
            self.expect(Token::Equal, "in variable declaration")?;

            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(VariableDeclaration::new(
            name,
            VariableType::Tensor(shape),
            initializer,
            location,
        ))
    }

    fn parse_type(&mut self) -> Result<Vec<u64>, Error> {
        self.expect(Token::AngleBracketOpen, "to begin type")?;

        let mut shape = vec![];

        while let Token::Number(number) = self.token {
            shape.push(number as u64);
            self.advance()?;

            if self.token == Token::Comma {
                self.advance()?;
            }
        }

        self.expect(Token::AngleBracketClose, "to end type")?;

        Ok(shape)
    }

    fn parse_expression(&mut self) -> Result<Expression, Error> {
        let lhs = self.parse_primary()?;

        self.parse_binary_rhs(0, lhs)
    }

    fn parse_binary_rhs(
        &mut self,
        precedence: usize,
        mut lhs: Expression,
    ) -> Result<Expression, Error> {
        loop {
            let Some((operator, token_precedence)) = self
                .binary_operator()
                .filter(|(_, token_precedence)| *token_precedence >= precedence)
            else {
                return Ok(lhs);
            };

            self.advance()?;

            // The tutorial attaches the location of the right-hand side to a binary operation.
            let location = self.location;
            let mut rhs = self.parse_primary()?;

            if self
                .binary_operator()
                .is_some_and(|(_, next_precedence)| token_precedence < next_precedence)
            {
                rhs = self.parse_binary_rhs(token_precedence + 1, rhs)?;
            }

            lhs = Expression::Binary(Binary::new(operator, lhs, rhs, location));
        }
    }

    const fn binary_operator(&self) -> Option<(BinaryOperator, usize)> {
        match self.token {
            Token::Minus => Some((BinaryOperator::Subtract, 20)),
            Token::Plus => Some((BinaryOperator::Add, 20)),
            Token::Star => Some((BinaryOperator::Multiply, 40)),
            Token::Dot => Some((BinaryOperator::Access, 60)),
            _ => None,
        }
    }

    fn parse_primary(&mut self) -> Result<Expression, Error> {
        match self.token {
            Token::Identifier(_) => self.parse_identifier_expression(),
            Token::Number(_) => Ok(Expression::Number(self.parse_number()?)),
            Token::ParenthesisOpen => self.parse_parenthesis_expression(),
            Token::BracketOpen => Ok(Expression::Literal(self.parse_tensor_literal()?)),
            Token::BraceOpen => Ok(Expression::StructLiteral(self.parse_struct_literal()?)),
            _ => self.fail("expression", "in primary expression"),
        }
    }

    fn parse_identifier_expression(&mut self) -> Result<Expression, Error> {
        let location = self.location;
        let name = self.parse_identifier("identifier", "in expression")?;

        if self.token != Token::ParenthesisOpen {
            return Ok(Expression::Variable(Variable::new(name, location)));
        } else if name == "print" {
            return Err(Error::Syntax {
                location,
                message: "print() is allowed only as a statement".into(),
            });
        }

        Ok(Expression::Call(Call::new(
            name,
            self.parse_arguments()?,
            location,
        )))
    }

    fn parse_arguments(&mut self) -> Result<Vec<Expression>, Error> {
        self.expect(Token::ParenthesisOpen, "in call expression")?;

        let mut arguments = vec![];

        if self.token != Token::ParenthesisClose {
            loop {
                arguments.push(self.parse_expression()?);

                if self.token == Token::ParenthesisClose {
                    break;
                } else if self.token != Token::Comma {
                    return self.fail("',' or ')'", "in argument list");
                }

                self.advance()?;
            }
        }

        self.expect(Token::ParenthesisClose, "in call expression")?;

        Ok(arguments)
    }

    fn parse_parenthesis_expression(&mut self) -> Result<Expression, Error> {
        self.expect(Token::ParenthesisOpen, "in parenthesized expression")?;

        let expression = self.parse_expression()?;

        self.expect(
            Token::ParenthesisClose,
            "to close expression with parentheses",
        )?;

        Ok(expression)
    }

    fn parse_number(&mut self) -> Result<Number, Error> {
        let Token::Number(value) = self.token else {
            return self.fail("number", "in expression");
        };
        let number = Number::new(value, self.location);

        self.advance()?;

        Ok(number)
    }

    fn parse_tensor_literal(&mut self) -> Result<Literal, Error> {
        let location = self.location;

        self.expect(Token::BracketOpen, "in literal expression")?;

        let mut values = vec![];

        loop {
            values.push(match self.token {
                Token::BracketOpen => Expression::Literal(self.parse_tensor_literal()?),
                Token::Number(_) => Expression::Number(self.parse_number()?),
                _ => return self.fail("<num> or [", "in literal expression"),
            });

            if self.token == Token::BracketClose {
                break;
            } else if self.token != Token::Comma {
                return self.fail("] or ,", "in literal expression");
            }

            self.advance()?;
        }

        self.expect(Token::BracketClose, "in literal expression")?;

        let mut dimensions = vec![values.len() as u64];

        if values
            .iter()
            .any(|value| matches!(value, Expression::Literal(_)))
        {
            let Some(Expression::Literal(first)) = values.first() else {
                return self.fail(
                    "uniform well-nested dimensions",
                    "inside literal expression",
                );
            };

            if !values.iter().all(|value| {
                matches!(value, Expression::Literal(literal) if literal.dimensions() == first.dimensions())
            }) {
                return self.fail(
                    "uniform well-nested dimensions",
                    "inside literal expression",
                );
            }

            dimensions.extend(first.dimensions());
        }

        Ok(Literal::new(values, dimensions, location))
    }

    fn parse_struct_literal(&mut self) -> Result<StructLiteral, Error> {
        let location = self.location;

        self.expect(Token::BraceOpen, "in struct literal expression")?;

        let mut values = vec![];

        loop {
            values.push(match self.token {
                Token::BracketOpen => Expression::Literal(self.parse_tensor_literal()?),
                Token::Number(_) => Expression::Number(self.parse_number()?),
                Token::BraceOpen => Expression::StructLiteral(self.parse_struct_literal()?),
                _ => return self.fail("{, [, or number", "in struct literal expression"),
            });

            if self.token == Token::BraceClose {
                break;
            } else if self.token != Token::Comma {
                return self.fail("} or ,", "in struct literal expression");
            }

            self.advance()?;
        }

        self.expect(Token::BraceClose, "in struct literal expression")?;

        Ok(StructLiteral::new(values, location))
    }

    fn parse_struct(&mut self) -> Result<Struct, Error> {
        let location = self.location;

        self.expect(Token::Struct, "in struct definition")?;

        let name = self.parse_identifier("name", "in struct definition")?;

        self.expect(Token::BraceOpen, "in struct definition")?;

        let mut variables = vec![];

        loop {
            variables.push(self.parse_declaration(false)?);

            self.expect(Token::Semicolon, "after variable in struct definition")?;

            if self.token == Token::BraceClose {
                break;
            }
        }

        self.expect(Token::BraceClose, "in struct definition")?;

        Ok(Struct::new(name, variables, location))
    }

    fn parse_identifier(&mut self, expected: &str, context: &str) -> Result<String, Error> {
        let Token::Identifier(name) = &self.token else {
            return self.fail(expected, context);
        };
        let name = name.clone();

        self.advance()?;

        Ok(name)
    }

    fn expect(&mut self, token: Token, context: &str) -> Result<(), Error> {
        if self.token != token {
            return self.fail(&token.to_string(), context);
        }

        self.advance()
    }

    fn advance(&mut self) -> Result<(), Error> {
        (self.token, self.location) = self.lexer.next_token()?;

        Ok(())
    }

    fn fail<T>(&self, expected: &str, context: &str) -> Result<T, Error> {
        Err(Error::Syntax {
            location: self.location,
            message: format!("expected {expected} {context} but found {}", self.token),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    fn parse_source(source: &str) -> Module {
        parse(source, "test.toy").unwrap()
    }

    #[test]
    fn parse_empty_module() {
        assert_eq!(parse_source(""), Module::new("test.toy".into(), vec![]));
    }

    #[test]
    fn parse_function_with_parameters() {
        assert_eq!(
            parse_source("def foo(a, Bar b) {}"),
            Module::new(
                "test.toy".into(),
                vec![Record::Function(Function::new(
                    Prototype::new(
                        "foo".into(),
                        vec![
                            VariableDeclaration::new(
                                "a".into(),
                                VariableType::Tensor(vec![]),
                                None,
                                Location::new(1, 9)
                            ),
                            VariableDeclaration::new(
                                "b".into(),
                                VariableType::Struct("Bar".into()),
                                None,
                                Location::new(1, 12)
                            ),
                        ],
                        Location::new(1, 1)
                    ),
                    vec![]
                ))]
            )
        );
    }

    #[test]
    fn parse_variable_declaration_with_shape() {
        assert_eq!(
            parse_source("def main() { var a<2, 3> = [1, 2, 3, 4, 5, 6]; }")
                .records()
                .first(),
            Some(&Record::Function(Function::new(
                Prototype::new("main".into(), vec![], Location::new(1, 1)),
                vec![Statement::VariableDeclaration(VariableDeclaration::new(
                    "a".into(),
                    VariableType::Tensor(vec![2, 3]),
                    Some(Expression::Literal(Literal::new(
                        (1..=6)
                            .map(|index| Expression::Number(Number::new(
                                index as f64,
                                Location::new(1, 29 + 3 * (index as usize - 1))
                            )))
                            .collect(),
                        vec![6],
                        Location::new(1, 28)
                    ))),
                    Location::new(1, 14)
                ))]
            )))
        );
    }

    #[test]
    fn parse_nested_literal() {
        assert_eq!(
            parse_source("def main() { var a = [[1, 2], [3, 4]]; }"),
            Module::new(
                "test.toy".into(),
                vec![Record::Function(Function::new(
                    Prototype::new("main".into(), vec![], Location::new(1, 1)),
                    vec![Statement::VariableDeclaration(VariableDeclaration::new(
                        "a".into(),
                        VariableType::Tensor(vec![]),
                        Some(Expression::Literal(Literal::new(
                            vec![
                                Expression::Literal(Literal::new(
                                    vec![
                                        Expression::Number(Number::new(1.0, Location::new(1, 24))),
                                        Expression::Number(Number::new(2.0, Location::new(1, 27))),
                                    ],
                                    vec![2],
                                    Location::new(1, 23)
                                )),
                                Expression::Literal(Literal::new(
                                    vec![
                                        Expression::Number(Number::new(3.0, Location::new(1, 32))),
                                        Expression::Number(Number::new(4.0, Location::new(1, 35))),
                                    ],
                                    vec![2],
                                    Location::new(1, 31)
                                )),
                            ],
                            vec![2, 2],
                            Location::new(1, 22)
                        ))),
                        Location::new(1, 14)
                    ))]
                ))]
            )
        );
    }

    #[test]
    fn parse_binary_operator_precedence() {
        // The location of a binary operation is the one of its right-hand side.
        assert_eq!(
            parse_source("def main() { return a + b * c; }"),
            Module::new(
                "test.toy".into(),
                vec![Record::Function(Function::new(
                    Prototype::new("main".into(), vec![], Location::new(1, 1)),
                    vec![Statement::Return(Return::new(
                        Some(Expression::Binary(Binary::new(
                            BinaryOperator::Add,
                            Expression::Variable(Variable::new("a".into(), Location::new(1, 21))),
                            Expression::Binary(Binary::new(
                                BinaryOperator::Multiply,
                                Expression::Variable(Variable::new(
                                    "b".into(),
                                    Location::new(1, 25)
                                )),
                                Expression::Variable(Variable::new(
                                    "c".into(),
                                    Location::new(1, 29)
                                )),
                                Location::new(1, 29)
                            )),
                            Location::new(1, 25)
                        ))),
                        Location::new(1, 14)
                    ))]
                ))]
            )
        );
    }

    #[test]
    fn parse_struct_access() {
        assert_eq!(
            parse_source("def main() { return x.a * 2; }"),
            Module::new(
                "test.toy".into(),
                vec![Record::Function(Function::new(
                    Prototype::new("main".into(), vec![], Location::new(1, 1)),
                    vec![Statement::Return(Return::new(
                        Some(Expression::Binary(Binary::new(
                            BinaryOperator::Multiply,
                            Expression::Binary(Binary::new(
                                BinaryOperator::Access,
                                Expression::Variable(Variable::new(
                                    "x".into(),
                                    Location::new(1, 21)
                                )),
                                Expression::Variable(Variable::new(
                                    "a".into(),
                                    Location::new(1, 23)
                                )),
                                Location::new(1, 23)
                            )),
                            Expression::Number(Number::new(2.0, Location::new(1, 27))),
                            Location::new(1, 27)
                        ))),
                        Location::new(1, 14)
                    ))]
                ))]
            )
        );
    }

    #[test]
    fn parse_struct_definition_and_literal() {
        assert_eq!(
            parse_source(indoc!(
                "
                struct Foo {
                  var a;
                  var b;
                }

                def main() {
                  Foo x = {1, [2, 3]};
                  print(x.a);
                }
                "
            )),
            Module::new(
                "test.toy".into(),
                vec![
                    Record::Struct(Struct::new(
                        "Foo".into(),
                        vec![
                            VariableDeclaration::new(
                                "a".into(),
                                VariableType::Tensor(vec![]),
                                None,
                                Location::new(2, 3)
                            ),
                            VariableDeclaration::new(
                                "b".into(),
                                VariableType::Tensor(vec![]),
                                None,
                                Location::new(3, 3)
                            ),
                        ],
                        Location::new(1, 1)
                    )),
                    Record::Function(Function::new(
                        Prototype::new("main".into(), vec![], Location::new(6, 1)),
                        vec![
                            Statement::VariableDeclaration(VariableDeclaration::new(
                                "x".into(),
                                VariableType::Struct("Foo".into()),
                                Some(Expression::StructLiteral(StructLiteral::new(
                                    vec![
                                        Expression::Number(Number::new(1.0, Location::new(7, 12))),
                                        Expression::Literal(Literal::new(
                                            vec![
                                                Expression::Number(Number::new(
                                                    2.0,
                                                    Location::new(7, 16)
                                                )),
                                                Expression::Number(Number::new(
                                                    3.0,
                                                    Location::new(7, 19)
                                                )),
                                            ],
                                            vec![2],
                                            Location::new(7, 15)
                                        )),
                                    ],
                                    Location::new(7, 11)
                                ))),
                                Location::new(7, 3)
                            )),
                            Statement::Print(Print::new(
                                Expression::Binary(Binary::new(
                                    BinaryOperator::Access,
                                    Expression::Variable(Variable::new(
                                        "x".into(),
                                        Location::new(8, 9)
                                    )),
                                    Expression::Variable(Variable::new(
                                        "a".into(),
                                        Location::new(8, 11)
                                    )),
                                    Location::new(8, 11)
                                )),
                                Location::new(8, 3)
                            )),
                        ]
                    )),
                ]
            )
        );
    }

    #[test]
    fn parse_empty_return() {
        assert_eq!(
            parse_source("def main() { return; }"),
            Module::new(
                "test.toy".into(),
                vec![Record::Function(Function::new(
                    Prototype::new("main".into(), vec![], Location::new(1, 1)),
                    vec![Statement::Return(Return::new(None, Location::new(1, 14)))]
                ))]
            )
        );
    }

    #[test]
    fn fail_to_parse_non_uniform_literal() {
        assert!(matches!(
            parse("def main() { var a = [[1, 2], [3]]; }", "test.toy"),
            Err(Error::Syntax { message, .. })
                if message.starts_with("expected uniform well-nested dimensions")
        ));
    }

    #[test]
    fn fail_to_parse_missing_semicolon() {
        assert!(matches!(
            parse("def main() { var a = 1 }", "test.toy"),
            Err(Error::Syntax { location, message })
                if location == Location::new(1, 24) && message == "expected ';' after expression but found '}'"
        ));
    }

    #[test]
    fn fail_to_parse_print_in_expression() {
        assert!(matches!(
            parse("def main() { var a = print(1); }", "test.toy"),
            Err(Error::Syntax { location, .. }) if location == Location::new(1, 22)
        ));
    }

    #[test]
    fn fail_to_parse_unknown_record() {
        assert!(matches!(
            parse("foo", "test.toy"),
            Err(Error::Syntax { location, .. }) if location == Location::new(1, 1)
        ));
    }
}
