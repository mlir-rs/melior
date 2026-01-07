use super::IdentifierList;
use proc_macro2::Ident;
use syn::{
    LitStr, Result, Token, bracketed,
    parse::{Parse, ParseStream},
};

pub struct PassSet {
    prefix: LitStr,
    identifiers: IdentifierList,
}

impl PassSet {
    pub const fn prefix(&self) -> &LitStr {
        &self.prefix
    }

    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}

impl Parse for PassSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let prefix = input.parse()?;
        <Token![,]>::parse(input)?;

        Ok(Self {
            prefix,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}
