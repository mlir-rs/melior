use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::Ident;

#[derive(Debug, Eq, PartialEq)]
pub enum TypeArgument {
    Set,
    Unset,
    Parameter(Ident),
}

impl TypeArgument {
    pub const fn parameter(&self) -> Option<&Ident> {
        if let Self::Parameter(identifier) = self {
            Some(identifier)
        } else {
            None
        }
    }
}

impl ToTokens for TypeArgument {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self {
            Self::Set => quote!(::melior::dialect::ods::__private::Set),
            Self::Unset => quote!(::melior::dialect::ods::__private::Unset),
            Self::Parameter(identifier) => quote!(#identifier),
        });
    }
}
