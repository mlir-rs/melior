use crate::location::Location;
use core::{
    error,
    fmt::{self, Display, Formatter},
    str::Utf8Error,
};

/// A compilation error.
#[derive(Debug)]
pub enum Error {
    /// A mismatched number of function arguments or results.
    ArgumentCount {
        /// A function name.
        function: String,
        /// An expected count.
        expected: usize,
        /// An actual count.
        actual: usize,
    },
    /// A failure of MLIR generation.
    CodeGeneration {
        /// A location of an AST node.
        location: Location,
        /// A message.
        message: String,
    },
    /// A function not found.
    FunctionNotFound(String),
    /// A function without a body block or a return operation.
    InvalidFunction(String),
    /// A failure to lower an operation.
    Lowering(String),
    /// A Melior error.
    Melior(melior::Error),
    /// A failure to parse an MLIR module.
    ModuleParse,
    /// A recursive function that cannot be inlined.
    RecursiveFunction(String),
    /// A failure of shape inference.
    ShapeInference(String),
    /// A syntax error.
    Syntax {
        /// A location of a token.
        location: Location,
        /// A message.
        message: String,
    },
    /// A failure of module verification.
    Verification,
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ArgumentCount {
                function,
                expected,
                actual,
            } => write!(
                formatter,
                "function '{function}' expects {expected} values but got {actual}"
            ),
            Self::CodeGeneration { location, message } => {
                write!(formatter, "{location}: {message}")
            }
            Self::FunctionNotFound(name) => write!(formatter, "function '{name}' not found"),
            Self::InvalidFunction(name) => write!(formatter, "function '{name}' is invalid"),
            Self::Lowering(operation) => {
                write!(formatter, "failed to lower operation: {operation}")
            }
            Self::Melior(error) => write!(formatter, "{error}"),
            Self::ModuleParse => write!(formatter, "failed to parse MLIR module"),
            Self::RecursiveFunction(name) => {
                write!(formatter, "recursive function '{name}' cannot be inlined")
            }
            Self::ShapeInference(message) => write!(formatter, "shape inference failed: {message}"),
            Self::Syntax { location, message } => write!(formatter, "{location}: {message}"),
            Self::Verification => write!(formatter, "module verification error"),
        }
    }
}

impl error::Error for Error {}

impl From<melior::Error> for Error {
    fn from(error: melior::Error) -> Self {
        Self::Melior(error)
    }
}

impl From<Utf8Error> for Error {
    fn from(error: Utf8Error) -> Self {
        Self::Melior(error.into())
    }
}
