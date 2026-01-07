use proc_macro2::Ident;
use syn::{
    Result, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub struct IdentifierList {
    identifiers: Vec<Ident>,
}

impl IdentifierList {
    pub fn identifiers(&self) -> &[Ident] {
        &self.identifiers
    }
}

impl Parse for IdentifierList {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            identifiers: Punctuated::<Ident, Token![,]>::parse_terminated(input)?
                .into_iter()
                .collect(),
        })
    }
}
