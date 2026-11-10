use crate::dialect::operation::{Attribute, OperationField};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_attribute_accessors(attribute: &Attribute) -> TokenStream {
    let getter = generate_getter(attribute);
    let setter = generate_setter(attribute);
    let remover = generate_remover(attribute);

    quote! {
        #getter
        #setter
        #remover
    }
}

fn generate_getter(attribute: &Attribute) -> TokenStream {
    let name = attribute.name();

    let identifier = attribute.singular_identifier();
    let return_type = attribute.return_type();
    let doc = format!("Returns the `{name}` attribute.");

    quote! {
        #[doc = #doc]
        #[allow(clippy::needless_question_mark)]
        pub fn #identifier(&self) -> #return_type {
            Ok(self.operation.attribute(#name)?.try_into()?)
        }
    }
}

fn generate_setter(attribute: &Attribute) -> TokenStream {
    let name = attribute.name();
    let identifier = attribute.set_identifier();
    let r#type = attribute.parameter_type();
    let doc = format!("Sets the `{name}` attribute.");

    quote! {
        #[doc = #doc]
        pub fn #identifier(&mut self, value: #r#type) {
            self.operation.set_attribute(#name, value.into());
        }
    }
}

fn generate_remover(attribute: &Attribute) -> Option<TokenStream> {
    if attribute.is_optional() {
        let name = attribute.name();
        let identifier = attribute.remove_identifier();
        let doc = format!("Removes the `{name}` attribute.");

        Some(quote! {
            #[doc = #doc]
            pub fn #identifier(&mut self) -> Result<(), ::melior::Error> {
                self.operation.remove_attribute(#name)
            }
        })
    } else {
        None
    }
}
