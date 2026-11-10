use super::IdentifierList;
use proc_macro2::Ident;
use syn::{
    Result, Token, bracketed,
    parse::{Parse, ParseStream},
};

pub struct IdentifierSet<T> {
    head: T,
    identifiers: IdentifierList,
}

impl<T> IdentifierSet<T> {
    pub const fn head(&self) -> &T {
        &self.head
    }

    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}

impl<T: Parse> Parse for IdentifierSet<T> {
    fn parse(input: ParseStream) -> Result<Self> {
        let head = input.parse()?;
        <Token![,]>::parse(input)?;

        Ok(Self {
            head,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}
