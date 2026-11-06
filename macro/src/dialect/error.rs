mod ods;

pub use self::ods::OdsError;
use core::{
    error,
    fmt::{self, Display, Formatter},
};
use proc_macro::TokenStream;
use proc_macro2::Span;
use tblgen::{
    SourceInfo,
    error::{SourceError, TableGenError},
};

#[derive(Debug)]
pub enum Error {
    Format(fmt::Error),
    InvalidIdentifier(String),
    Ods(SourceError<OdsError>),
    Parse(tblgen::Error),
    Syn(syn::Error),
    TableGen(tblgen::Error),
}

impl Error {
    pub fn add_source_info(self, info: SourceInfo) -> Self {
        match self {
            Self::TableGen(error) => error.add_source_info(info).into(),
            Self::Ods(error) => error.add_source_info(info).into(),
            Self::Parse(error) => Self::Parse(error.add_source_info(info)),
            Self::Format(_) | Self::InvalidIdentifier(_) | Self::Syn(_) => self,
        }
    }

    pub fn to_compile_error(&self) -> TokenStream {
        match self {
            Self::Syn(err) => err.to_compile_error().into(),
            Self::TableGen(_)
            | Self::Ods(_)
            | Self::Parse(_)
            | Self::Format(_)
            | Self::InvalidIdentifier(_) => syn::Error::new(Span::call_site(), self)
                .to_compile_error()
                .into(),
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self {
            Self::Format(error) => write!(formatter, "{error}"),
            Self::InvalidIdentifier(identifier) => {
                write!(formatter, "invalid identifier: {identifier}")
            }
            Self::Ods(error) => write!(formatter, "invalid ODS input: {error}"),
            Self::Parse(error) => write!(formatter, "failed to parse TableGen source: {error}"),
            Self::Syn(error) => write!(formatter, "failed to parse macro input: {error}"),
            Self::TableGen(error) => write!(formatter, "invalid ODS input: {error}"),
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

impl From<fmt::Error> for Error {
    fn from(error: fmt::Error) -> Self {
        Self::Format(error)
    }
}
