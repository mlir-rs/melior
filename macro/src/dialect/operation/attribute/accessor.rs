use super::super::Attribute;
use crate::dialect::operation::operation_field::OperationFieldV2;
use crate::dialect::{error::Error, utility::sanitize_snake_case_name};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_accessors(attribute: &Attribute) -> Result<TokenStream, Error> {
    let getter = generate_getter(attribute)?;
    let setter = generate_setter(attribute)?;
    let remover = generate_remover(attribute)?;

    Ok(quote! {
        #getter
        #setter
        #remover
    })
}

fn generate_getter(attribute: &Attribute) -> Result<TokenStream, Error> {
    let name = attribute.name();

    let ident = attribute.sanitized_name();
    let return_type = attribute.return_type();
    let body = if attribute.constraint().is_unit() {
        quote! { self.operation.attribute(#name).is_some() }
    } else {
        // TODO Handle returning `melior::Attribute`.
        quote! { Ok(self.operation.attribute(#name)?.try_into()?) }
    };

    Ok(quote! {
        #[allow(clippy::needless_question_mark)]
        pub fn #ident(&self, context: &'c ::melior::Context) -> #return_type {
            #body
        }
    })
}

fn generate_setter(attribute: &Attribute) -> Result<TokenStream, Error> {
    let name = attribute.name();

    let body = if attribute.constraint().is_unit() {
        quote! {
            if value {
                self.operation.set_attribute(#name, Attribute::unit(&self.operation.context()));
            } else {
                self.operation.remove_attribute(#name)
            }
        }
    } else {
        quote! {
            self.operation.set_attribute(#name, &value.into());
        }
    };

    let ident = sanitize_snake_case_name(&format!("set_{}", attribute.name()))?;
    let r#type = attribute.parameter_type();

    Ok(quote! {
        pub fn #ident(&mut self, context: &'c ::melior::Context, value: #r#type) {
            #body
        }
    })
}

fn generate_remover(attribute: &Attribute) -> Result<Option<TokenStream>, Error> {
    let constrait = attribute.constraint();

    Ok(if constrait.is_unit() || constrait.is_optional() {
        let name = attribute.name();
        let ident = sanitize_snake_case_name(&format!("remove_{}", attribute.name()))?;

        Some(quote! {
            pub fn #ident(&mut self, context: &'c ::melior::Context) -> Result<(), ::melior::Error> {
                self.operation.remove_attribute(#name)
            }
        })
    } else {
        None
    })
}
