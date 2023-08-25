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
    Io(io::Error),
    Ods(SourceError<OdsError>),
    Parse(tblgen::Error),
    Syn(syn::Error),
    TableGen(tblgen::Error),
    Utf8(FromUtf8Error),
}

impl Error {
    pub fn add_source_info(self, info: SourceInfo) -> Self {
        match self {
            Self::TableGen(error) => error.add_source_info(info).into(),
            Self::Ods(error) => error.add_source_info(info).into(),
            Self::Parse(error) => Self::Parse(error.add_source_info(info)),
            Self::Io(_) | Self::Syn(_) | Self::Utf8(_) => self,
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Ods(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::Parse(error) => write!(formatter, "failed to parse TableGen source: {error}"),
            Self::Syn(error) => write!(formatter, "failed to parse macro input: {error}"),
            Self::TableGen(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::Utf8(error) => write!(formatter, "{error}"),
        }
    }
}

impl error::Error for Error {}

impl From<SourceError<OdsError>> for Error {
    fn from(error: SourceError<OdsError>) -> Self {
        Self::Ods(error)
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
pub enum OdsError {
    ExpectedSuperClass(&'static str),
    InvalidTrait,
}

impl Display for OdsError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::ExpectedSuperClass(class) => write!(
                formatter,
                "expected this record to be a subclass of {class}",
            ),
            Self::InvalidTrait => write!(formatter, "record is not a supported trait"),
        }
    }
}

impl error::Error for OdsError {}
