use std::{
    convert::Infallible,
    error,
    fmt::{self, Display, Formatter},
    str::Utf8Error,
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    AttributeExpected(&'static str, String),
    AttributeNotFound(String),
    AttributeParse(String),
    BlockArgumentExpected(String),
    ElementExpected {
        r#type: &'static str,
        value: String,
    },
    InvokeFunction,
    OperationBuild,
    OperandNotFound(&'static str),
    OperationResultExpected(String),
    PositionOutOfBounds {
        name: &'static str,
        value: String,
        index: usize,
    },
    ParsePassPipeline(String),
    ResultNotFound(&'static str),
    RunPass,
    TypeExpected(&'static str, String),
    UnknownDiagnosticSeverity(u32),
    Utf8(Utf8Error),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::AttributeExpected(r#type, attribute) => {
                write!(formatter, "{type} attribute expected: {attribute}")
            }
            Self::AttributeNotFound(name) => {
                write!(formatter, "attribute {name} not found")
            }
            Self::AttributeParse(string) => {
                write!(formatter, "failed to parse attribute: {string}")
            }
            Self::BlockArgumentExpected(value) => {
                write!(formatter, "block argument expected: {value}")
            }
            Self::ElementExpected { r#type, value } => {
                write!(formatter, "element of {type} type expected: {value}")
            }
            Self::InvokeFunction => write!(formatter, "failed to invoke JIT-compiled function"),
            Self::OperationBuild => {
                write!(formatter, "operation build failed")
            }
            Self::OperandNotFound(name) => {
                write!(formatter, "operand {name} not found")
            }
            Self::OperationResultExpected(value) => {
                write!(formatter, "operation result expected: {value}")
            }
            Self::ParsePassPipeline(message) => {
                write!(formatter, "failed to parse pass pipeline:\n{message}")
            }
            Self::PositionOutOfBounds { name, value, index } => {
                write!(formatter, "{name} position {index} out of bounds: {value}")
            }
            Self::ResultNotFound(name) => {
                write!(formatter, "result {name} not found")
            }
            Self::RunPass => write!(formatter, "failed to run pass"),
            Self::TypeExpected(r#type, actual) => {
                write!(formatter, "{type} type expected: {actual}")
            }
            Self::UnknownDiagnosticSeverity(severity) => {
                write!(formatter, "unknown diagnostic severity: {severity}")
            }
            Self::Utf8(error) => {
                write!(formatter, "{error}")
            }
        }
    }
}

impl error::Error for Error {}

impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Self::Utf8(error)
    }
}

impl From<Infallible> for Error {
    fn from(_: Infallible) -> Self {
        unreachable!()
    }
}
