mod input_field;

use self::input_field::InputField;
use std::ops::Deref;
use syn::{Token, parse::Parse, punctuated::Punctuated};

pub struct DialectInput {
    name: String,
    // TODO Remove this field.
    include_directories: Vec<String>,
    files: Vec<String>,
    directories: Vec<String>,
}

impl DialectInput {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn include_directories(&self) -> impl Iterator<Item = &str> {
        self.include_directories.iter().map(Deref::deref)
    }

    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(Deref::deref)
    }

    pub fn directories(&self) -> impl Iterator<Item = &str> {
        self.directories.iter().map(Deref::deref)
    }
}

impl Parse for DialectInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut includes = vec![];
        let mut files = vec![];
        let mut directories = vec![];

        for item in Punctuated::<InputField, Token![,]>::parse_terminated(input)? {
            match item {
                InputField::Name(field) => name = Some(field.value()),
                InputField::IncludeDirectories(field) => {
                    includes = field.into_iter().map(|literal| literal.value()).collect()
                }
                InputField::Files(field) => {
                    files = field.into_iter().map(|literal| literal.value()).collect()
                }
                InputField::Directories(field) => {
                    directories = field.into_iter().map(|literal| literal.value()).collect()
                }
            }
        }

        Ok(Self {
            name: name.ok_or_else(|| input.error("dialect name required"))?,
            include_directories: includes,
            files,
            directories,
        })
    }
}
