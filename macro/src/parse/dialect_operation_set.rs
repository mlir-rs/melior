use super::IdentifierList;
use proc_macro2::Ident;
use syn::{
    Result, Token, bracketed,
    parse::{Parse, ParseStream},
};

pub struct DialectOperationSet {
    dialect: Ident,
    identifiers: IdentifierList,
}

impl DialectOperationSet {
    pub const fn dialect(&self) -> &Ident {
        &self.dialect
    }

    pub fn identifiers(&self) -> &[Ident] {
        self.identifiers.identifiers()
    }
}

impl Parse for DialectOperationSet {
    fn parse(input: ParseStream) -> Result<Self> {
        let dialect = Ident::parse(input)?;
        <Token![,]>::parse(input)?;

        Ok(Self {
            dialect,
            identifiers: {
                let content;
                bracketed!(content in input);
                content.parse::<IdentifierList>()?
            },
        })
    }
}
