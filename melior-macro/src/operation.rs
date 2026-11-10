use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{format_ident, quote};
use syn::{Type, parse_quote};

pub fn generate_binary(dialect: &Ident, names: &[Ident]) -> TokenStream {
    generate(
        dialect,
        names,
        &format_ident!("binary_operator"),
        &[
            (format_ident!("lhs"), parse_quote!(crate::ir::Value<'c, '_>)),
            (format_ident!("rhs"), parse_quote!(crate::ir::Value<'c, '_>)),
        ],
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
        &[(
            format_ident!("value"),
            parse_quote!(crate::ir::Value<'c, '_>),
        )],
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
        &[
            (
                format_ident!("value"),
                parse_quote!(crate::ir::Value<'c, '_>),
            ),
            (format_ident!("r#type"), parse_quote!(crate::ir::Type<'c>)),
        ],
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
    parameters: &[(Ident, Type)],
    builder: proc_macro2::TokenStream,
) -> TokenStream {
    let documents = names.iter().map(|name| create_document(dialect, name));
    let operation_names = names
        .iter()
        .map(|name| create_operation_name(dialect, name));
    let parameter_names = parameters.iter().map(|(name, _)| name);
    let arguments = quote! { #(#parameter_names),* };
    let parameters = parameters
        .iter()
        .map(|(name, r#type)| quote! { #name: #r#type, })
        .collect::<proc_macro2::TokenStream>();

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
