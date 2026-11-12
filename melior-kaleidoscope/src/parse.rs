//! A parser.

use crate::{
    ast::{
        Binary, BinaryOperator, Call, Expression, Function, Literal, Module, Number, Print,
        Prototype, Record, Return, Statement, Struct, StructLiteral, Type, Variable,
        VariableDeclaration, VariableDefinition,
    },
    error,
    location::Location,
};
use winnow::{
    ModalResult, Parser,
    ascii::{dec_uint, digit1, till_line_ending},
    combinator::{
        self, Infix, alt, cut_err, delimited, dispatch, empty, eof, fail, opt, peek, preceded,
        repeat, separated, terminated,
    },
    error::{AddContext, ContextError, ErrMode, StrContext, StrContextValue},
    stream::{LocatingSlice, Location as _, Stateful, Stream},
    token::{any, one_of, take_while},
};

const KEYWORDS: &[&str] = &["def", "print", "return", "struct", "var"];

// The state is the byte offsets of line starts to compute locations.
type Input<'a> = Stateful<LocatingSlice<&'a str>, &'a [usize]>;

/// Parses a module.
pub fn parse(source: &str, path: &str) -> Result<Module, error::Error> {
    let line_starts = [0]
        .into_iter()
        .chain(source.match_indices('\n').map(|(index, _)| index + 1))
        .collect::<Vec<_>>();

    module
        .parse(Stateful {
            input: LocatingSlice::new(source),
            state: &line_starts,
        })
        .map(|records| Module::new(path.into(), records))
        .map_err(|error| error::Error::Syntax {
            location: location_at(&line_starts, error.offset()),
            message: match error.inner().to_string() {
                message if message.is_empty() => "invalid syntax".into(),
                message => message,
            },
        })
}

fn location_at(line_starts: &[usize], offset: usize) -> Location {
    let line = line_starts.partition_point(|&start| start <= offset);

    Location::new(line, offset - line_starts[line - 1] + 1)
}

fn module(input: &mut Input<'_>) -> ModalResult<Vec<Record>> {
    terminated(
        repeat(0.., record),
        token(eof).context(StrContext::Expected(StrContextValue::Description(
            "function or struct definition",
        ))),
    )
    .parse_next(input)
}

fn record(input: &mut Input<'_>) -> ModalResult<Record> {
    token(alt((
        function.map(Record::Function),
        r#struct.map(Record::Struct),
    )))
    .parse_next(input)
}

fn function(input: &mut Input<'_>) -> ModalResult<Function> {
    (prototype, cut_err(block))
        .map(|(prototype, body)| Function::new(prototype, body))
        .parse_next(input)
}

fn prototype(input: &mut Input<'_>) -> ModalResult<Prototype> {
    (
        located(keyword("def")),
        cut_err((
            identifier,
            delimited(
                symbol('('),
                separated(0.., parameter, symbol(',')),
                symbol(')'),
            ),
        )),
    )
        .map(|((location, _), (name, parameters))| Prototype::new(name, parameters, location))
        .parse_next(input)
}

// A parameter is an optional struct type name followed by a name.
fn parameter(input: &mut Input<'_>) -> ModalResult<VariableDeclaration> {
    located((opt(terminated(identifier, peek(identifier))), identifier))
        .map(|(location, (type_name, name))| {
            VariableDeclaration::new(
                name,
                if let Some(name) = type_name {
                    Type::Struct(name)
                } else {
                    Type::Tensor(vec![])
                },
                location,
            )
        })
        .parse_next(input)
}

fn r#struct(input: &mut Input<'_>) -> ModalResult<Struct> {
    (
        located(keyword("struct")),
        cut_err((
            identifier,
            delimited(
                symbol('{'),
                repeat(1.., terminated(variable_declaration, symbol(';'))),
                symbol('}'),
            ),
        )),
    )
        .map(|((location, _), (name, variables))| Struct::new(name, variables, location))
        .parse_next(input)
}

fn block(input: &mut Input<'_>) -> ModalResult<Vec<Statement>> {
    delimited(
        symbol('{'),
        preceded(
            semicolons,
            repeat(
                0..,
                terminated(statement, (cut_err(symbol(';')), semicolons)),
            ),
        ),
        cut_err(symbol('}')),
    )
    .parse_next(input)
}

fn semicolons(input: &mut Input<'_>) -> ModalResult<()> {
    repeat(0.., symbol(';')).parse_next(input)
}

fn statement(input: &mut Input<'_>) -> ModalResult<Statement> {
    token(alt((
        print_statement.map(Statement::Print),
        return_statement.map(Statement::Return),
        variable_definition.map(Statement::VariableDefinition),
        expression.map(Statement::Expression),
    )))
    .parse_next(input)
}

fn print_statement(input: &mut Input<'_>) -> ModalResult<Print> {
    (
        located(keyword("print")),
        cut_err(delimited(symbol('('), expression, symbol(')'))),
    )
        .map(|((location, _), argument)| Print::new(argument, location))
        .parse_next(input)
}

fn return_statement(input: &mut Input<'_>) -> ModalResult<Return> {
    (located(keyword("return")), opt(expression))
        .map(|((location, _), value)| Return::new(value, location))
        .parse_next(input)
}

// A variable declaration is either a `var` declaration with an optional shape
// or a declaration with a struct type name.
fn variable_declaration(input: &mut Input<'_>) -> ModalResult<VariableDeclaration> {
    alt((
        (located(keyword("var")), cut_err((identifier, opt(shape)))).map(
            |((location, _), (name, shape))| {
                VariableDeclaration::new(name, Type::Tensor(shape.unwrap_or_default()), location)
            },
        ),
        (located(identifier), identifier).map(|((location, type_name), name)| {
            VariableDeclaration::new(name, Type::Struct(type_name), location)
        }),
    ))
    .parse_next(input)
}

fn variable_definition(input: &mut Input<'_>) -> ModalResult<VariableDefinition> {
    (
        variable_declaration,
        cut_err(preceded(symbol('='), expression)),
    )
        .map(|(declaration, initializer)| VariableDefinition::new(declaration, initializer))
        .parse_next(input)
}

fn shape(input: &mut Input<'_>) -> ModalResult<Vec<u64>> {
    delimited(
        symbol('<'),
        cut_err(separated(0.., token(dec_uint::<_, u64, _>), symbol(','))),
        cut_err(symbol('>')),
    )
    .parse_next(input)
}

fn expression(input: &mut Input<'_>) -> ModalResult<Expression> {
    label(
        token(combinator::expression(primary).infix(token(dispatch! {any;
            '+' => empty.map(|()| Infix::Left(20, |_, lhs, rhs| Ok(binary(BinaryOperator::Add, lhs, rhs)))),
            '-' => empty.map(|()| Infix::Left(20, |_, lhs, rhs| Ok(binary(BinaryOperator::Subtract, lhs, rhs)))),
            '*' => empty.map(|()| Infix::Left(40, |_, lhs, rhs| Ok(binary(BinaryOperator::Multiply, lhs, rhs)))),
            '.' => empty.map(|()| Infix::Left(60, |_, lhs, rhs| Ok(binary(BinaryOperator::Access, lhs, rhs)))),
            _ => fail,
        }))),
        "expression",
    )
    .parse_next(input)
}

// The tutorial attaches the location of the first token of a right-hand side
// to a binary operation.
fn binary(operator: BinaryOperator, lhs: Expression, rhs: Expression) -> Expression {
    let location = first_location(&rhs);

    Expression::Binary(Binary::new(operator, lhs, rhs, location))
}

fn first_location(expression: &Expression) -> Location {
    match expression {
        Expression::Binary(binary) => first_location(binary.lhs()),
        _ => expression.location(),
    }
}

fn primary(input: &mut Input<'_>) -> ModalResult<Expression> {
    alt((
        call_or_variable,
        number.map(Expression::Number),
        delimited(symbol('('), expression, cut_err(symbol(')'))),
        tensor_literal.map(Expression::Literal),
        struct_literal.map(Expression::StructLiteral),
    ))
    .parse_next(input)
}

fn call_or_variable(input: &mut Input<'_>) -> ModalResult<Expression> {
    let (location, name) = located(identifier).parse_next(input)?;

    Ok(if let Some(arguments) = opt(arguments).parse_next(input)? {
        Expression::Call(Call::new(name, arguments, location))
    } else {
        Expression::Variable(Variable::new(name, location))
    })
}

fn arguments(input: &mut Input<'_>) -> ModalResult<Vec<Expression>> {
    delimited(
        symbol('('),
        cut_err(separated(0.., expression, symbol(','))),
        cut_err(symbol(')')),
    )
    .parse_next(input)
}

fn tensor_literal(input: &mut Input<'_>) -> ModalResult<Literal> {
    let (location, values): (_, Vec<_>) = located(delimited(
        symbol('['),
        cut_err(separated(
            1..,
            alt((
                tensor_literal.map(Expression::Literal),
                number.map(Expression::Number),
            )),
            symbol(','),
        )),
        cut_err(symbol(']')),
    ))
    .parse_next(input)?;
    let mut dimensions = vec![values.len() as u64];

    // Dimensions of nested literals must be uniform.
    if let Some(first) = values.iter().find_map(|value| match value {
        Expression::Literal(literal) => Some(literal.dimensions()),
        _ => None,
    }) {
        if !values.iter().all(
            |value| matches!(value, Expression::Literal(literal) if literal.dimensions() == first),
        ) {
            return cut_err(
                fail.context(StrContext::Expected(StrContextValue::Description(
                    "uniform well-nested dimensions",
                ))),
            )
            .parse_next(input);
        }

        dimensions.extend(first);
    }

    Ok(Literal::new(values, dimensions, location))
}

fn struct_literal(input: &mut Input<'_>) -> ModalResult<StructLiteral> {
    located(delimited(
        symbol('{'),
        cut_err(separated(
            1..,
            alt((
                tensor_literal.map(Expression::Literal),
                number.map(Expression::Number),
                struct_literal.map(Expression::StructLiteral),
            )),
            symbol(','),
        )),
        cut_err(symbol('}')),
    ))
    .map(|(location, values)| StructLiteral::new(values, location))
    .parse_next(input)
}

fn number(input: &mut Input<'_>) -> ModalResult<Number> {
    located(
        (
            digit1,
            take_while(0.., |character: char| {
                character.is_ascii_digit() || character == '.'
            }),
        )
            .take()
            .try_map(str::parse),
    )
    .map(|(location, value)| Number::new(value, location))
    .parse_next(input)
}

fn identifier(input: &mut Input<'_>) -> ModalResult<String> {
    token(word.verify(|word: &str| !KEYWORDS.contains(&word)))
        .map(String::from)
        .context(StrContext::Expected(StrContextValue::Description(
            "identifier",
        )))
        .parse_next(input)
}

fn keyword<'a>(keyword: &'static str) -> impl Parser<Input<'a>, &'a str, ErrMode<ContextError>> {
    token(word.verify(move |word: &str| word == keyword)).context(StrContext::Expected(
        StrContextValue::StringLiteral(keyword),
    ))
}

fn word<'a>(input: &mut Input<'a>) -> ModalResult<&'a str> {
    (
        one_of(|character: char| character.is_ascii_alphabetic()),
        take_while(0.., |character: char| {
            character.is_ascii_alphanumeric() || character == '_'
        }),
    )
        .take()
        .parse_next(input)
}

fn symbol<'a>(symbol: char) -> impl Parser<Input<'a>, char, ErrMode<ContextError>> {
    token(symbol).context(StrContext::Expected(StrContextValue::CharLiteral(symbol)))
}

// Attaches a location of a first character of a token to its output.
fn located<'a, O>(
    parser: impl Parser<Input<'a>, O, ErrMode<ContextError>>,
) -> impl Parser<Input<'a>, (Location, O), ErrMode<ContextError>> {
    token((location, parser))
}

fn location(input: &mut Input<'_>) -> ModalResult<Location> {
    Ok(location_at(input.state, input.current_token_start()))
}

fn token<'a, O>(
    parser: impl Parser<Input<'a>, O, ErrMode<ContextError>>,
) -> impl Parser<Input<'a>, O, ErrMode<ContextError>> {
    preceded(whitespace, parser)
}

// Describes what is expected on a backtracking error. Committed errors keep
// their own descriptions.
fn label<'a, O>(
    mut parser: impl Parser<Input<'a>, O, ErrMode<ContextError>>,
    description: &'static str,
) -> impl Parser<Input<'a>, O, ErrMode<ContextError>> {
    move |input: &mut Input<'a>| {
        let start = input.checkpoint();

        parser.parse_next(input).map_err(|error| match error {
            ErrMode::Backtrack(error) => ErrMode::Backtrack(error.add_context(
                input,
                &start,
                StrContext::Expected(StrContextValue::Description(description)),
            )),
            error => error,
        })
    }
}

// Comments start with `#` and last until the ends of lines.
fn whitespace(input: &mut Input<'_>) -> ModalResult<()> {
    repeat(
        0..,
        alt((
            take_while(1.., char::is_whitespace).void(),
            ('#', till_line_ending).void(),
        )),
    )
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    fn parse_source(source: &str) -> Module {
        parse(source, "test.toy").unwrap()
    }

    #[test]
    fn parse_empty_module() {
        assert_eq!(
            parse_source("# comment only\n"),
            Module::new("test.toy".into(), vec![])
        );
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
                                Type::Tensor(vec![]),
                                Location::new(1, 9)
                            ),
                            VariableDeclaration::new(
                                "b".into(),
                                Type::Struct("Bar".into()),
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
                vec![Statement::VariableDefinition(VariableDefinition::new(
                    VariableDeclaration::new(
                        "a".into(),
                        Type::Tensor(vec![2, 3]),
                        Location::new(1, 14)
                    ),
                    Expression::Literal(Literal::new(
                        (1..=6)
                            .map(|index| Expression::Number(Number::new(
                                index as f64,
                                Location::new(1, 29 + 3 * (index as usize - 1))
                            )))
                            .collect(),
                        vec![6],
                        Location::new(1, 28)
                    ))
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
                    vec![Statement::VariableDefinition(VariableDefinition::new(
                        VariableDeclaration::new(
                            "a".into(),
                            Type::Tensor(vec![]),
                            Location::new(1, 14)
                        ),
                        Expression::Literal(Literal::new(
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
                        ))
                    ))]
                ))]
            )
        );
    }

    #[test]
    fn parse_binary_operator_precedence() {
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
                                Type::Tensor(vec![]),
                                Location::new(2, 3)
                            ),
                            VariableDeclaration::new(
                                "b".into(),
                                Type::Tensor(vec![]),
                                Location::new(3, 3)
                            ),
                        ],
                        Location::new(1, 1)
                    )),
                    Record::Function(Function::new(
                        Prototype::new("main".into(), vec![], Location::new(6, 1)),
                        vec![
                            Statement::VariableDefinition(VariableDefinition::new(
                                VariableDeclaration::new(
                                    "x".into(),
                                    Type::Struct("Foo".into()),
                                    Location::new(7, 3)
                                ),
                                Expression::StructLiteral(StructLiteral::new(
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
                                ))
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
                if message == "expected uniform well-nested dimensions"
        ));
    }

    #[test]
    fn fail_to_parse_missing_semicolon() {
        assert!(matches!(
            parse("def main() { var a = 1 }", "test.toy"),
            Err(Error::Syntax { location, message })
                if location == Location::new(1, 24) && message == "expected `;`"
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

    #[test]
    fn fail_to_parse_invalid_number() {
        assert!(matches!(
            parse("def main() { var a = 1.2.3; }", "test.toy"),
            Err(Error::Syntax { location, .. }) if location == Location::new(1, 22)
        ));
    }
}
