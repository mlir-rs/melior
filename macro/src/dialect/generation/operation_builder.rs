use crate::dialect::operation::{
    Attribute, OperationBuilder, OperationElement, OperationField, TypeInference,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub fn generate_operation_builder(builder: &OperationBuilder) -> TokenStream {
    let result_fns = match builder.operation().type_inference() {
        Some(_) => Default::default(),
        None => builder
            .operation()
            .results()
            .map(|result| generate_field_fn(builder, result))
            .collect::<Vec<_>>(),
    };
    let infer_from_operands = matches!(
        builder.operation().type_inference(),
        Some(TypeInference::SameOperands)
    );
    let operand_fns = builder
        .operation()
        .operands()
        .enumerate()
        .map(|(i, operand)| {
            if i == 0 && infer_from_operands {
                generate_same_operands_first_fn(builder, operand)
            } else {
                generate_field_fn(builder, operand)
            }
        })
        .collect::<Vec<_>>();
    let region_fns = builder
        .operation()
        .regions()
        .map(|region| generate_field_fn(builder, region))
        .collect::<Vec<_>>();
    let successor_fns = builder
        .operation()
        .successors()
        .map(|successor| generate_field_fn(builder, successor))
        .collect::<Vec<_>>();
    let infer_from_first_attr = matches!(
        builder.operation().type_inference(),
        Some(TypeInference::FirstAttrDerived)
    );
    let attribute_fns = builder
        .operation()
        .attributes()
        .enumerate()
        .map(|(i, attribute)| {
            if i == 0 && infer_from_first_attr {
                generate_first_attr_derived_fn(builder, attribute)
            } else {
                generate_field_fn(builder, attribute)
            }
        })
        .collect::<Vec<_>>();

    let new_fn = generate_new_fn(builder);
    let build_fn = generate_build_fn(builder);

    let identifier = builder.identifier();
    let doc = format!(
        "A builder for {}.",
        builder.operation().documentation_name()
    );
    let type_parameters = builder.type_state().parameters().collect::<Vec<_>>();

    quote! {
        #[doc = #doc]
        pub struct #identifier<'c, #(#type_parameters),*> {
            builder: ::melior::ir::operation::OperationBuilder<'c>,
            context: &'c ::melior::Context,
            _state: ::std::marker::PhantomData<(#(#type_parameters),*)>,
        }

        #new_fn

        #(#result_fns)*
        #(#operand_fns)*
        #(#region_fns)*
        #(#successor_fns)*
        #(#attribute_fns)*

        #build_fn
    }
}

// TODO Split this function for different kinds of fields.
fn generate_field_fn(builder: &OperationBuilder, field: &impl OperationField) -> TokenStream {
    let builder_identifier = builder.identifier();
    let identifier = field.singular_identifier();
    let parameter_type = field.parameter_type();
    let argument = quote! { #identifier: #parameter_type };
    let add_identifier = format_ident!("add_{}", field.plural_kind_identifier());

    // Argument types can be singular and variadic. But `add` functions in Melior
    // are always variadic, so we need to create a slice or `Vec` for singular
    // arguments.
    let add_arguments = field.add_arguments(identifier);

    if field.is_optional() {
        let parameters = builder.type_state().parameters().collect::<Vec<_>>();

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    self.builder = self.builder.#add_identifier(#add_arguments);
                    self
                }
            }
        }
    } else {
        let parameters = builder.type_state().parameters_without(field.name());
        let arguments_set = builder.type_state().arguments_with(field.name(), true);
        let arguments_unset = builder.type_state().arguments_with(field.name(), false);

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    #builder_identifier {
                        context: self.context,
                        builder: self.builder.#add_identifier(#add_arguments),
                        _state: Default::default(),
                    }
                }
            }
        }
    }
}

// Mirrors C++'s genUseOperandAsResultTypeSeparateParamBuilder. Intentionally a
// sibling of generate_field_fn rather than merged into it, matching the C++
// structure where these are also separate functions.
fn generate_same_operands_first_fn(
    builder: &OperationBuilder,
    field: &impl OperationElement,
) -> TokenStream {
    let builder_identifier = builder.identifier();
    let identifier = field.singular_identifier();
    let parameter_type = field.parameter_type();
    let argument = quote! { #identifier: #parameter_type };
    let add_identifier = format_ident!("add_{}", field.plural_kind_identifier());
    let add_arguments = field.add_arguments(identifier);
    let result_count = builder.operation().result_len();
    let result_type_copies: Vec<_> = (0..result_count).map(|_| quote! { result_type }).collect();
    // For variadic operands the parameter is `&[Value]`; index into it for the
    // type. For singular operands the parameter is `Value`; take a reference
    // directly.
    let type_access = if field.is_variadic() {
        quote! { ::melior::ir::ValueLike::r#type(&#identifier[0]) }
    } else {
        quote! { ::melior::ir::ValueLike::r#type(&#identifier) }
    };

    if field.is_optional() {
        let parameters = builder.type_state().parameters().collect::<Vec<_>>();
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    let result_type = #type_access;
                    self.builder = self.builder
                        .add_results(&[#(#result_type_copies),*])
                        .#add_identifier(#add_arguments);
                    self
                }
            }
        }
    } else {
        let parameters = builder.type_state().parameters_without(field.name());
        let arguments_set = builder.type_state().arguments_with(field.name(), true);
        let arguments_unset = builder.type_state().arguments_with(field.name(), false);
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    let result_type = #type_access;
                    #builder_identifier {
                        context: self.context,
                        builder: self.builder
                            .add_results(&[#(#result_type_copies),*])
                            .#add_identifier(#add_arguments),
                        _state: Default::default(),
                    }
                }
            }
        }
    }
}

// Mirrors C++'s genUseAttrAsResultTypeBuilder. Intentionally a sibling of
// generate_field_fn for the same reason as generate_same_operands_first_fn.
fn generate_first_attr_derived_fn(builder: &OperationBuilder, field: &Attribute) -> TokenStream {
    let builder_identifier = builder.identifier();
    let identifier = field.singular_identifier();
    let parameter_type = field.parameter_type();
    let argument = quote! { #identifier: #parameter_type };
    let add_arguments = field.add_arguments(identifier);
    let result_count = builder.operation().result_len();
    let result_type_copies: Vec<_> = (0..result_count).map(|_| quote! { result_type }).collect();
    // If the attribute is a TypeAttr, use its wrapped type; otherwise use the
    // attribute's own type.
    let type_access = if field.is_type() {
        quote! { #identifier.value() }
    } else {
        quote! { ::melior::ir::attribute::AttributeLike::r#type(&#identifier) }
    };

    if field.is_optional() {
        let parameters = builder.type_state().parameters().collect::<Vec<_>>();
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    let result_type = #type_access;
                    self.builder = self.builder
                        .add_results(&[#(#result_type_copies),*])
                        .add_attributes(#add_arguments);
                    self
                }
            }
        }
    } else {
        let parameters = builder.type_state().parameters_without(field.name());
        let arguments_set = builder.type_state().arguments_with(field.name(), true);
        let arguments_unset = builder.type_state().arguments_with(field.name(), false);
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    let result_type = #type_access;
                    #builder_identifier {
                        context: self.context,
                        builder: self.builder
                            .add_results(&[#(#result_type_copies),*])
                            .add_attributes(#add_arguments),
                        _state: Default::default(),
                    }
                }
            }
        }
    }
}

fn generate_build_fn(builder: &OperationBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let arguments = builder.type_state().arguments_with_all(true);
    let operation_identifier = format_ident!("{}", &builder.operation().name());
    let error = format!("should be a valid {operation_identifier}");
    let maybe_infer = matches!(
        builder.operation().type_inference(),
        Some(TypeInference::Interface)
    )
    .then_some(quote! { .enable_result_type_inference() });

    quote! {
        impl<'c> #identifier<'c, #(#arguments),*> {
            pub fn build(self) -> #operation_identifier<'c> {
                self.builder #maybe_infer.build().expect("valid operation").try_into().expect(#error)
            }
        }
    }
}

fn generate_new_fn(builder: &OperationBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let name = &builder.operation().full_operation_name();
    let arguments = builder.type_state().arguments_with_all(false);

    quote! {
        impl<'c> #identifier<'c, #(#arguments),*> {
            pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
                Self {
                    context,
                    builder: ::melior::ir::operation::OperationBuilder::new(#name, location),
                    _state: Default::default(),
                }
            }
        }
    }
}

pub fn generate_operation_builder_fn(builder: &OperationBuilder) -> TokenStream {
    let builder_ident = builder.identifier();
    let arguments = builder.type_state().arguments_with_all(false);

    quote! {
        /// Creates a builder.
        pub fn builder(
            context: &'c ::melior::Context,
            location: ::melior::ir::Location<'c>
        ) -> #builder_ident<'c, #(#arguments),*> {
            #builder_ident::new(context, location)
        }
    }
}

pub fn generate_default_constructor(builder: &OperationBuilder) -> TokenStream {
    let operation_identifier = format_ident!("{}", &builder.operation().name());
    let constructor_identifier = builder.operation().constructor_identifier();
    let arguments = builder
        .operation()
        .required_fields()
        .map(|field| {
            let r#type = &field.parameter_type();
            let name = &field.singular_identifier();

            quote! { #name: #r#type }
        })
        .chain([quote! { location: ::melior::ir::Location<'c> }])
        .collect::<Vec<_>>();
    let builder_calls = builder
        .operation()
        .required_fields()
        .map(|field| {
            let name = &field.singular_identifier();

            quote! { .#name(#name) }
        })
        .collect::<Vec<_>>();

    let doc = format!("Creates {}.", builder.operation().documentation_name());

    quote! {
        #[allow(clippy::too_many_arguments)]
        #[doc = #doc]
        pub fn #constructor_identifier<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #operation_identifier<'c> {
            #operation_identifier::builder(context, location)#(#builder_calls)*.build()
        }
    }
}
