use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

#[derive(Debug)]
pub enum OdsError {
    ExpectedSuperClass(&'static str),
    InvalidTrait,
    UnexpectedSuperClass(&'static str),
    UnnamedDagArg(&'static str),
}

impl Display for OdsError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ExpectedSuperClass(class) => {
                write!(formatter, "record should be a sub-class of {class}",)
            }
            Self::InvalidTrait => write!(formatter, "record is not a supported trait"),
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
