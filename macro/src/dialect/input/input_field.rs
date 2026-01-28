use proc_macro2::Ident;
use quote::format_ident;
use syn::{LitStr, Token, bracketed, parse::Parse, punctuated::Punctuated};

pub enum InputField {
    Name(LitStr),
    Files(Punctuated<LitStr, Token![,]>),
    Directories(Punctuated<LitStr, Token![,]>),
}

impl Parse for InputField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<Ident>()?;

        input.parse::<Token![:]>()?;

        if ident == format_ident!("name") {
            Ok(Self::Name(input.parse()?))
        } else if ident == format_ident!("files") {
            let content;
            bracketed!(content in input);
            Ok(Self::Files(
                Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?,
            ))
        } else if ident == format_ident!("include_directories") {
            let content;
            bracketed!(content in input);
            Ok(Self::Directories(
                Punctuated::<LitStr, Token![,]>::parse_terminated(&content)?,
            ))
        } else {
            Err(input.error(format!("invalid field {ident}")))
        }
    }
}
