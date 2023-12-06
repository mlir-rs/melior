mod type_state_item;
mod type_state_list;

use self::{type_state_item::TypeStateItem, type_state_list::TypeStateList};
use super::{
    super::{error::Error, utility::sanitize_snake_case_name},
    FieldKind, Operation, OperationField,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};

pub fn generate_operation_builder(builder: &OperationBuilder) -> Result<TokenStream, Error> {
    let field_names = builder
        .type_state
        .field_names()
        .map(sanitize_snake_case_name)
        .collect::<Result<Vec<_>, _>>()?;

    let phantom_fields = builder
        .type_state
        .parameters()
        .zip(&field_names)
        .map(|(r#type, name)| {
            quote! {
                #name: ::std::marker::PhantomData<#r#type>
            }
        });

    let phantom_arguments = field_names
        .iter()
        .map(|name| quote! { #name: ::std::marker::PhantomData })
        .collect::<Vec<_>>();

    let builder_fns = builder
        .create_builder_fns(&field_names, phantom_arguments.as_slice())
        .collect::<Result<Vec<_>, _>>()?;

    let new = builder.create_new_fn(phantom_arguments.as_slice())?;
    let build = builder.create_build_fn()?;

    let builder_identifier = builder.builder_identifier()?;
    let doc = format!("Builder for {}", builder.operation.summary()?);
    let iter_arguments = builder.type_state.parameters();

    Ok(quote! {
        #[doc = #doc]
        pub struct #builder_identifier<'c, #(#iter_arguments),*> {
            builder: ::melior::ir::operation::OperationBuilder<'c>,
            context: &'c ::melior::Context,
            #(#phantom_fields),*
        }

        #new

        #(#builder_fns)*

        #build
    })
}

pub struct OperationBuilder<'o> {
    operation: &'o Operation<'o>,
    type_state: TypeStateList,
}

impl<'o> OperationBuilder<'o> {
    pub fn new(operation: &'o Operation<'o>) -> Result<Self, Error> {
        Ok(Self {
            operation,
            type_state: Self::create_type_state(operation)?,
        })
    }

    pub fn create_builder_fns<'a>(
        &'a self,
        field_names: &'a [Ident],
        phantoms: &'a [TokenStream],
    ) -> impl Iterator<Item = Result<TokenStream, Error>> + 'a {
        self.operation.fields().map(move |field| {
            // TODO Initialize a builder identifier out of this closure.
            let builder_ident = self.builder_identifier()?;
            let name = sanitize_snake_case_name(field.name)?;
            let parameter_type = field.kind.parameter_type()?;
            let argument = quote! { #name: #parameter_type };
            let add = format_ident!("add_{}s", field.kind.as_str());

            // Argument types can be singular and variadic, but add functions in melior
            // are always variadic, so we need to create a slice or vec for singular
            // arguments
            let add_arguments = match &field.kind {
                FieldKind::Element { constraint, .. } => {
                    if constraint.has_unfixed() && !constraint.is_optional() {
                        quote! { #name }
                    } else {
                        quote! { &[#name] }
                    }
                }
                FieldKind::Attribute { .. } => {
                    let name_string = &field.name;

                    quote! {
                        &[(
                            ::melior::ir::Identifier::new(self.context, #name_string),
                            #name.into(),
                        )]
                    }
                }
                FieldKind::Successor { constraint, .. } => {
                    if constraint.is_variadic() {
                        quote! { #name }
                    } else {
                        quote! { &[#name] }
                    }
                }
                FieldKind::Region { constraint, .. } => {
                    if constraint.is_variadic() {
                        quote! { #name }
                    } else {
                        quote! { vec![#name] }
                    }
                }
            };

            Ok(if field.kind.is_optional()? {
                let parameters = self.type_state.parameters().collect::<Vec<_>>();

                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#parameters),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#parameters),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            self
                        }
                    }
                }
            } else if field.kind.is_result() && self.operation.can_infer_type {
                quote!()
            } else {
                let parameters = self.type_state.parameters_without(field.name);
                let arguments_set = self.type_state.arguments_set(field.name, true);
                let arguments_unset = self.type_state.arguments_set(field.name, false);

                quote! {
                    impl<'c, #(#parameters),*> #builder_ident<'c, #(#arguments_unset),*> {
                        pub fn #name(mut self, #argument) -> #builder_ident<'c, #(#arguments_set),*> {
                            self.builder = self.builder.#add(#add_arguments);
                            let Self { context, mut builder, #(#field_names),* } = self;
                            #builder_ident {
                                context,
                                builder,
                                #(#phantoms),*
                            }
                        }
                    }
                }
            })
        })
    }

    fn create_build_fn(&self) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let arguments = self.type_state.arguments_all_set(true);
        let class_name = format_ident!("{}", &self.operation.class_name()?);
        let error = format!("should be a valid {class_name}");
        let maybe_infer = self
            .operation
            .can_infer_type
            .then_some(quote! { .enable_result_type_inference() });

        Ok(quote! {
            impl<'c> #builder_ident<'c, #(#arguments),*> {
                pub fn build(self) -> #class_name<'c> {
                    self.builder #maybe_infer.build().expect("valid operation").try_into().expect(#error)
                }
            }
        })
    }

    fn create_new_fn(&self, phantoms: &[TokenStream]) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let name = &self.operation.full_name()?;
        let arguments = self.type_state.arguments_all_set(false);

        Ok(quote! {
            impl<'c> #builder_ident<'c, #(#arguments),*> {
                pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
                    Self {
                        context,
                        builder: ::melior::ir::operation::OperationBuilder::new( #name, location),
                        #(#phantoms),*
                    }
                }
            }
        })
    }

    pub fn create_op_builder_fn(&self) -> Result<TokenStream, Error> {
        let builder_ident = self.builder_identifier()?;
        let arguments = self.type_state.arguments_all_set(false);

        Ok(quote! {
            pub fn builder(
                context: &'c ::melior::Context,
                location: ::melior::ir::Location<'c>
            ) -> #builder_ident<'c, #(#arguments),*> {
                #builder_ident::new(context, location)
            }
        })
    }

    pub fn create_default_constructor(&self) -> Result<TokenStream, Error> {
        let class_name = format_ident!("{}", &self.operation.class_name()?);
        let name = sanitize_snake_case_name(self.operation.short_name()?)?;
        let arguments = Self::required_fields(self.operation)
            .map(|field| {
                let field = field?;
                let parameter_type = &field.kind.parameter_type()?;
                let parameter_name = &field.sanitized_name;

                Ok(quote! { #parameter_name: #parameter_type })
            })
            .chain([Ok(quote! { location: ::melior::ir::Location<'c> })])
            .collect::<Result<Vec<_>, Error>>()?;
        let builder_calls = Self::required_fields(self.operation)
            .map(|field| {
                let parameter_name = &field?.sanitized_name;

                Ok(quote! { .#parameter_name(#parameter_name) })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let doc = format!("Creates a new {}", self.operation.summary()?);

        Ok(quote! {
            #[allow(clippy::too_many_arguments)]
            #[doc = #doc]
            pub fn #name<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #class_name<'c> {
                #class_name::builder(context, location)#(#builder_calls)*.build()
            }
        })
    }

    fn required_fields<'a, 'b>(
        operation: &'a Operation<'b>,
    ) -> impl Iterator<Item = Result<&'a OperationField<'b>, Error>> {
        operation
            .fields()
            .filter(|field| !field.kind.is_result() || !operation.can_infer_type)
            .filter_map(|field| match field.kind.is_optional() {
                Ok(optional) => (!optional).then_some(Ok(field)),
                Err(error) => Some(Err(error)),
            })
    }

    fn create_type_state(operation: &'o Operation<'o>) -> Result<TypeStateList, Error> {
        Ok(TypeStateList::new(
            Self::required_fields(operation)
                .enumerate()
                .map(|(index, field)| Ok(TypeStateItem::new(index, field?.name.to_string())))
                .collect::<Result<_, Error>>()?,
        ))
    }

    fn builder_identifier(&self) -> Result<Ident, Error> {
        Ok(format_ident!("{}Builder", self.operation.class_name()?))
    }
}
