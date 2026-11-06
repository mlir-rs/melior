use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{format_ident, quote};

pub fn generate_binary(dialect: &Ident, names: &[Ident]) -> TokenStream {
    generate(
        dialect,
        names,
        &format_ident!("binary_operator"),
        quote! {
            lhs: crate::ir::Value<'c, '_>,
            rhs: crate::ir::Value<'c, '_>,
        },
        quote! { lhs, rhs },
        quote! {
            .add_operands(&[lhs, rhs])
            .enable_result_type_inference()
        },
    )
}

pub fn generate_unary(dialect: &Ident, names: &[Ident]) -> TokenStream {
    generate(
        dialect,
        names,
        &format_ident!("unary_operator"),
        quote! {
            value: crate::ir::Value<'c, '_>,
        },
        quote! { value },
        quote! {
            .add_operands(&[value])
            .enable_result_type_inference()
        },
    )
}

pub fn generate_typed_unary(dialect: &Ident, names: &[Ident]) -> TokenStream {
    generate(
        dialect,
        names,
        &format_ident!("typed_unary_operator"),
        quote! {
            value: crate::ir::Value<'c, '_>,
            r#type: crate::ir::Type<'c>,
        },
        quote! { value, r#type },
        quote! {
            .add_operands(&[value])
            .add_results(&[r#type])
        },
    )
}

fn generate(
    dialect: &Ident,
    names: &[Ident],
    helper: &Ident,
    parameters: proc_macro2::TokenStream,
    arguments: proc_macro2::TokenStream,
    builder: proc_macro2::TokenStream,
) -> TokenStream {
    let documents = names.iter().map(|name| create_document(dialect, name));
    let operation_names = names
        .iter()
        .map(|name| create_operation_name(dialect, name));

    quote! {
        #(
            #[doc = #documents]
            pub fn #names<'c>(
                #parameters
                location: crate::ir::Location<'c>,
            ) -> crate::ir::Operation<'c> {
                #helper(#operation_names, #arguments, location)
            }
        )*

        fn #helper<'c>(
            name: &str,
            #parameters
            location: crate::ir::Location<'c>,
        ) -> crate::ir::Operation<'c> {
            crate::ir::operation::OperationBuilder::new(name, location)
                #builder
                .build()
                .expect("valid operation")
        }
    }
    .into()
}

fn create_document(dialect: &Ident, name: &Ident) -> String {
    format!(
        " Creates an `{}` operation.",
        create_operation_name(dialect, name)
    )
}

fn create_operation_name(dialect: &Ident, name: &Ident) -> String {
    format!("{dialect}.{name}")
}
