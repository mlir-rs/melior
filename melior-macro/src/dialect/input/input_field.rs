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
        let identifier = input.parse::<Ident>()?;

        input.parse::<Token![:]>()?;

        match identifier.to_string().as_str() {
            "name" => Ok(Self::Name(input.parse()?)),
            "files" => Ok(Self::Files(parse_string_list(input)?)),
            "include_directories" => Ok(Self::Directories(parse_string_list(input)?)),
            "include_directory_env_vars" => Ok(Self::DirectoryEnvVars(parse_string_list(input)?)),
            _ => Err(input.error(format!("invalid field {identifier}"))),
        }
    }
}

fn parse_string_list(input: syn::parse::ParseStream) -> syn::Result<Punctuated<LitStr, Token![,]>> {
    let content;
    bracketed!(content in input);
    Punctuated::parse_terminated(&content)
}
