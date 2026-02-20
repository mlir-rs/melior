mod input_field;

use self::input_field::InputField;
use std::ops::Deref;
use proc_macro2::Span;
use syn::{Token, parse::Parse, punctuated::Punctuated};

pub struct DialectInput {
    name: String,
    files: Vec<String>,
    directories: Vec<String>,
    directory_env_vars: Vec<(String, Span)>,
}

impl DialectInput {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(Deref::deref)
    }

    pub fn directories(&self) -> impl Iterator<Item = &str> {
        self.directories.iter().map(Deref::deref)
    }

    pub fn directory_env_vars(&self) -> impl Iterator<Item = (&str, &Span)> {
        self.directory_env_vars.iter().map(|(value, span)|{
            (value.deref(), span)
        })
    }
}

impl Parse for DialectInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut files = vec![];
        let mut directories = vec![];
        let mut directory_env_vars = vec![];

        for item in Punctuated::<InputField, Token![,]>::parse_terminated(input)? {
            match item {
                InputField::Name(field) => name = Some(field.value()),
                InputField::Files(field) => {
                    files = field.into_iter().map(|literal| literal.value()).collect()
                }
                InputField::Directories(field) => {
                    directories = field.into_iter().map(|literal| literal.value()).collect()
                }
                InputField::DirectoryEnvVars(field) => {
                    directory_env_vars = field.into_iter().map(|literal| (literal.value(), literal.span())).collect()
                }
            }
        }

        Ok(Self {
            name: name.ok_or_else(|| input.error("dialect name required"))?,
            files,
            directories,
            directory_env_vars,
        })
    }
}
