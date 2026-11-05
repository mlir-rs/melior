use proc_macro2::Ident;
use syn::{LitStr, Token, bracketed, parse::Parse, punctuated::Punctuated};

pub enum InputField {
    Name(LitStr),
    Files(Punctuated<LitStr, Token![,]>),
    Directories(Punctuated<LitStr, Token![,]>),
    DirectoryEnvVars(Punctuated<LitStr, Token![,]>),
}

impl Parse for InputField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<Ident>()?;

        input.parse::<Token![:]>()?;

        if ident == "name" {
            Ok(Self::Name(input.parse()?))
        } else if ident == "files" {
            Ok(Self::Files(parse_string_list(input)?))
        } else if ident == "include_directories" {
            Ok(Self::Directories(parse_string_list(input)?))
        } else if ident == "include_directory_env_vars" {
            Ok(Self::DirectoryEnvVars(parse_string_list(input)?))
        } else {
            Err(input.error(format!("invalid field {ident}")))
        }
    }
}

fn parse_string_list(input: syn::parse::ParseStream) -> syn::Result<Punctuated<LitStr, Token![,]>> {
    let content;
    bracketed!(content in input);
    Punctuated::parse_terminated(&content)
}
