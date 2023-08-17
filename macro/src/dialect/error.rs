use std::{
    error,
    fmt::{self, Display, Formatter},
    io,
    string::FromUtf8Error,
};
use tblgen::{
    error::{SourceError, TableGenError},
    SourceInfo,
};

#[derive(Debug)]
pub enum Error {
    ExpectedSuperClass(SourceError<ExpectedSuperClassError>),
    Io(io::Error),
    ParseError,
    Syn(syn::Error),
    TableGen(tblgen::Error),
    Utf8(FromUtf8Error),
}

impl Error {
    pub fn add_source_info(self, info: SourceInfo) -> Self {
        match self {
            Self::TableGen(error) => error.add_source_info(info).into(),
            Self::ExpectedSuperClass(error) => error.add_source_info(info).into(),
            Self::Io(_) | Self::ParseError | Self::Syn(_) | Self::Utf8(_) => self,
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ExpectedSuperClass(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::ParseError => write!(formatter, "error parsing TableGen source"),
            Self::Syn(error) => write!(formatter, "failed to parse macro input: {error}"),
            Self::TableGen(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::Utf8(error) => write!(formatter, "{error}"),
        }
    }
}

impl error::Error for Error {}

impl From<SourceError<ExpectedSuperClassError>> for Error {
    fn from(error: SourceError<ExpectedSuperClassError>) -> Self {
        Self::ExpectedSuperClass(error)
    }
}

impl From<SourceError<TableGenError>> for Error {
    fn from(error: SourceError<TableGenError>) -> Self {
        Self::TableGen(error)
    }
}

impl From<syn::Error> for Error {
    fn from(error: syn::Error) -> Self {
        Self::Syn(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<FromUtf8Error> for Error {
    fn from(error: FromUtf8Error) -> Self {
        Self::Utf8(error)
    }
}

#[derive(Debug)]
pub struct ExpectedSuperClassError(pub String);

impl Display for ExpectedSuperClassError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(
            formatter,
            "expected this record to be a subclass of {}",
            self.0
        )
    }
}

impl error::Error for ExpectedSuperClassError {}
