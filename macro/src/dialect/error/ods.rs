use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

#[derive(Debug)]
pub enum OdsError {
    ExpectedSuperClass(&'static str),
    GeneratedNameConflict(String),
    InvalidTrait,
    MissingVariadicSizeTrait,
    UnexpectedSuperClass(&'static str),
    UnnamedDagArg(&'static str),
}

impl Display for OdsError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ExpectedSuperClass(class) => {
                write!(formatter, "record should be a sub-class of {class}",)
            }
            Self::GeneratedNameConflict(name) => {
                write!(
                    formatter,
                    "generated name {name} for an unnamed result conflicts with a named element"
                )
            }
            Self::InvalidTrait => write!(formatter, "record is not a supported trait"),
            Self::MissingVariadicSizeTrait => write!(
                formatter,
                "multiple variadic operands/results require SameVariadicOperandSize, \
                SameVariadicResultSize, or AttrSizedOperandSegments/AttrSizedResultSegments trait"
            ),
            Self::UnexpectedSuperClass(class) => {
                write!(formatter, "record should not be a sub-class of {class}",)
            }
            Self::UnnamedDagArg(dag) => {
                write!(formatter, "unnamed argument in {dag} DAG")
            }
        }
    }
}

impl Error for OdsError {}
