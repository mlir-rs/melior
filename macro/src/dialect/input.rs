mod input_field;

use self::input_field::InputField;
use std::ops::Deref;
use syn::{Token, parse::Parse, punctuated::Punctuated};

pub struct DialectInput {
    name: String,
    // TODO Remove this field.
    table_gen: Option<String>,
    // TODO Remove this field.
    td_file: Option<String>,
    // TODO Remove this field.
    include_directories: Vec<String>,
    files: Vec<String>,
    directories: Vec<String>,
}

impl DialectInput {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn table_gen(&self) -> Option<&str> {
        self.table_gen.as_deref()
    }

    pub fn td_file(&self) -> Option<&str> {
        self.td_file.as_deref()
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
        let mut table_gen = None;
        let mut td_file = None;
        let mut includes = vec![];
        let mut files = vec![];
        let mut directories = vec![];

        for item in Punctuated::<InputField, Token![,]>::parse_terminated(input)? {
            match item {
                InputField::Name(field) => name = Some(field.value()),
                InputField::TableGen(td) => table_gen = Some(td.value()),
                InputField::TdFile(file) => td_file = Some(file.value()),
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
            table_gen,
            td_file,
            include_directories: includes,
            files,
            directories,
        })
    }
}
