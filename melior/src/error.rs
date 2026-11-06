use core::{
    convert::Infallible,
    error,
    fmt::{self, Display, Formatter},
    str::Utf8Error,
};

/// A Melior error.
#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    /// A failure of pattern application.
    ApplyPatterns,
    /// An attribute of a type expected.
    AttributeExpected(&'static str, String),
    /// An attribute not found.
    AttributeNotFound(String),
    /// A failure of attribute parsing.
    AttributeParse(String),
    /// A block argument expected.
    BlockArgumentExpected(String),
    /// An element of a type expected.
    ElementExpected {
        /// A type name.
        r#type: &'static str,
        /// A value.
        value: String,
    },
    /// A failure of function invocation.
    InvokeFunction,
    /// A failure of operation build.
    OperationBuild,
    /// An operand not found.
    OperandNotFound(&'static str),
    /// An operation of a name expected.
    OperationExpected {
        /// An expected operation name.
        expected: &'static str,
        /// An actual operation.
        actual: String,
    },
    /// An operation result expected.
    OperationResultExpected(String),
    /// A failure of pass pipeline parsing.
    ParsePassPipeline(String),
    /// A position out of bounds.
    PositionOutOfBounds {
        /// A name of positioned elements.
        name: &'static str,
        /// A value containing the elements.
        value: String,
        /// An index.
        index: usize,
    },
    /// A result not found.
    ResultNotFound(&'static str),
    /// A failure of a pass run.
    RunPass,
    /// A type expected.
    TypeExpected(&'static str, String),
    /// An unknown diagnostic severity.
    UnknownDiagnosticSeverity(u32),
    /// A UTF-8 error.
    Utf8(Utf8Error),
    /// Mismatched numbers of values.
    ValueCountMismatch {
        /// A number of values to replace.
        from: usize,
        /// A number of replacement values.
        to: usize,
    },
    /// A failure of bytecode writing.
    WriteBytecode,
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ApplyPatterns => write!(formatter, "failed to apply patterns"),
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
            Self::OperationExpected { expected, actual } => {
                write!(formatter, "expected operation {expected}, got {actual}")
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
            Self::ValueCountMismatch { from, to } => {
                write!(
                    formatter,
                    "value count mismatch: {from} from values, {to} to values"
                )
            }
            Self::WriteBytecode => write!(formatter, "failed to write bytecode"),
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
