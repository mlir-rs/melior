use crate::r#type::Type;
use std::error;
use std::fmt::Display;
use std::fmt::{self, Formatter};

#[derive(Debug, Eq, PartialEq)]
pub enum Error<'c> {
    FunctionExpected(Type<'c>),
}

impl<'c> Display for Error<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::FunctionExpected(r#type) => write!(formatter, "function expected: {}", r#type),
        }
    }
}

impl<'c> error::Error for Error<'c> {}
