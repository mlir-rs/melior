use super::type_state::TypeStateBuilder;
use crate::dialect::operation::{
    Attribute, FieldKind, Operation, OperationElement, OperationField, TypeInference,
};
use core::iter::repeat_n;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Type, ext::IdentExt};

pub fn generate_operation_builder(builder: &TypeStateBuilder) -> TokenStream {
    let result_fns = builder
        .operation()
        .non_inferred_results()
        .map(|result| generate_field_fn(builder, result))
        .collect::<Vec<_>>();
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
    let segment_fields = builder.operation().segment_arrays().map(|(kind, length)| {
        let identifier = kind.segment_size_field_identifier();

        quote! { #identifier: [i32; #length] }
    });

    quote! {
        #[doc = #doc]
        pub struct #identifier<'c, #(#type_parameters),*> {
            builder: ::melior::ir::operation::OperationBuilder<'c>,
            context: &'c ::melior::Context,
            #(#segment_fields,)*
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

// Generates the segment-size fields of a builder struct literal. The array
// holding the field's own group, if any, gets that group's length; every other
// array is copied forward.
fn generate_segment_size_fields(
    operation: &Operation,
    field: &impl OperationField,
) -> Vec<TokenStream> {
    operation
        .segment_arrays()
        .map(|(kind, _)| {
            let identifier = kind.segment_size_field_identifier();

            if let Some(index) = field.segment()
                && field.kind() == kind
            {
                let sizes = Ident::new("sizes", Span::mixed_site());
                let size = if matches!(field.parameter_type(), Type::Reference(_)) {
                    let identifier = field.singular_identifier();

                    quote! { #identifier.len() as i32 }
                } else {
                    quote! { 1 }
                };

                quote! {
                    #identifier: {
                        let mut #sizes = self.#identifier;
                        #sizes[#index] = #size;
                        #sizes
                    }
                }
            } else {
                quote! { #identifier: self.#identifier }
            }
        })
        .collect()
}

fn generate_setter(
    builder: &TypeStateBuilder,
    field: &impl OperationField,
    statements: TokenStream,
    calls: TokenStream,
) -> TokenStream {
    let builder_identifier = builder.identifier();
    let identifier = field.singular_identifier();
    let parameter_type = field.parameter_type();
    let argument = quote! { #identifier: #parameter_type };
    let doc = format!("Sets `{}`.", field.name());
    let kind = field.kind();
    let index = field.index();
    let successor = kind == FieldKind::Successor;
    let unsafety = successor.then(|| quote!(unsafe));
    let safety_doc = successor.then(generate_successor_safety_doc);
    let calls = if successor {
        quote! { unsafe { self.builder #calls } }
    } else {
        quote! { self.builder #calls }
    };

    if builder.type_state().has_parameter(kind, index) {
        let arguments_set = builder
            .type_state()
            .arguments_with(kind, index, true)
            .collect::<Vec<_>>();
        let arguments_unset = builder
            .type_state()
            .arguments_with(kind, index, false)
            .collect::<Vec<_>>();
        let parameters = arguments_unset
            .iter()
            .filter_map(|argument| argument.parameter());
        let segment_size_fields = generate_segment_size_fields(builder.operation(), field);
        let option_setter = field.is_optional().then(|| {
            let option_identifier = format_ident!("{}_option", identifier.unraw());
            let doc = format!("Sets `{}` if it is present.", field.name());
            let segment_size_fields = builder.operation().segment_arrays().map(|(kind, _)| {
                let identifier = kind.segment_size_field_identifier();

                quote! { #identifier: self.#identifier }
            });

            quote! {
                #[doc = #doc]
                pub fn #option_identifier(self, #identifier: Option<#parameter_type>) -> #builder_identifier<'c, #(#arguments_set),*> {
                    if let Some(#identifier) = #identifier {
                        self.#identifier(#identifier)
                    } else {
                        #builder_identifier {
                            context: self.context,
                            #(#segment_size_fields,)*
                            builder: self.builder,
                            _state: Default::default(),
                        }
                    }
                }
            }
        });

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                #[doc = #doc]
                #safety_doc
                pub #unsafety fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    #statements

                    #builder_identifier {
                        context: self.context,
                        #(#segment_size_fields,)*
                        builder: #calls,
                        _state: Default::default(),
                    }
                }

                #option_setter
            }
        }
    } else {
        let parameters = builder.type_state().parameters().collect::<Vec<_>>();

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                #[doc = #doc]
                #safety_doc
                pub #unsafety fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    #statements
                    self.builder = #calls;
                    self
                }
            }
        }
    }
}

fn generate_field_fn(builder: &TypeStateBuilder, field: &impl OperationField) -> TokenStream {
    let add_identifier = field.add_identifier();
    let add_arguments = field.add_arguments();

    generate_setter(
        builder,
        field,
        quote! {},
        quote! { .#add_identifier(#add_arguments) },
    )
}

// Mirrors C++'s genUseOperandAsResultTypeSeparateParamBuilder. Intentionally a
// sibling of generate_field_fn rather than merged into it, matching the C++
// structure where these are also separate functions.
fn generate_same_operands_first_fn(
    builder: &TypeStateBuilder,
    field: &OperationElement,
) -> TokenStream {
    let identifier = field.singular_identifier();
    let add_identifier = field.add_identifier();
    let add_arguments = field.add_arguments();
    let result_type_copies = repeat_n(quote! { result_type }, builder.operation().result_len());
    // For variadic operands the parameter is `&[Value]`; index into it for the
    // type. For singular operands the parameter is `Value`; take a reference
    // directly.
    let type_access = if field.is_variadic() {
        quote! { ::melior::ir::ValueLike::r#type(&#identifier[0]) }
    } else {
        quote! { ::melior::ir::ValueLike::r#type(&#identifier) }
    };

    generate_setter(
        builder,
        field,
        quote! { let result_type = #type_access; },
        quote! {
            .add_results(&[#(#result_type_copies),*])
            .#add_identifier(#add_arguments)
        },
    )
}

// Mirrors C++'s genUseAttrAsResultTypeBuilder. Intentionally a sibling of
// generate_field_fn for the same reason as generate_same_operands_first_fn.
fn generate_first_attr_derived_fn(builder: &TypeStateBuilder, field: &Attribute) -> TokenStream {
    let identifier = field.singular_identifier();
    let add_arguments = field.add_arguments();
    let result_type_copies = repeat_n(quote! { result_type }, builder.operation().result_len());
    // If the attribute is a TypeAttr, use its wrapped type; otherwise use the
    // attribute's own type.
    let type_access = if field.is_type() {
        quote! { #identifier.value() }
    } else {
        quote! { ::melior::ir::attribute::AttributeLike::r#type(&#identifier) }
    };

    generate_setter(
        builder,
        field,
        quote! { let result_type = #type_access; },
        quote! {
            .add_results(&[#(#result_type_copies),*])
            .add_attributes(#add_arguments)
        },
    )
}

fn generate_build_fn(builder: &TypeStateBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let arguments = builder.type_state().final_arguments().collect::<Vec<_>>();
    let parameters = arguments.iter().filter_map(|argument| argument.parameter());
    let operation_identifier = format_ident!("{}", &builder.operation().name());
    let error = format!("should be a valid {operation_identifier}");
    let maybe_infer = matches!(
        builder.operation().type_inference(),
        Some(TypeInference::Interface)
    )
    .then_some(quote! { .enable_result_type_inference() });
    let add_segment_size_attributes = builder.operation().segment_arrays().map(|(kind, _)| {
        let field_identifier = kind.segment_size_field_identifier();
        let name = kind.segment_size_attribute_name();

        quote! {
            .add_attributes(&[(
                ::melior::ir::Identifier::new(self.context, #name),
                ::melior::ir::attribute::DenseI32ArrayAttribute::new(
                    self.context,
                    &self.#field_identifier,
                ).into(),
            )])
        }
    });

    quote! {
        impl<'c, #(#parameters),*> #identifier<'c, #(#arguments),*> {
            /// Builds an operation.
            pub fn build(self) -> #operation_identifier<'c> {
                self.builder #(#add_segment_size_attributes)* #maybe_infer
                    .build().expect("valid operation").try_into().expect(#error)
            }
        }
    }
}

fn generate_new_fn(builder: &TypeStateBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let name = &builder.operation().full_operation_name();
    let arguments = builder.type_state().initial_arguments();
    let segment_fields = builder.operation().segment_arrays().map(|(kind, length)| {
        let identifier = kind.segment_size_field_identifier();

        quote! { #identifier: [0; #length] }
    });

    quote! {
        impl<'c> #identifier<'c, #(#arguments),*> {
            /// Creates a builder.
            pub fn new(context: &'c ::melior::Context, location: ::melior::ir::Location<'c>) -> Self {
                Self {
                    context,
                    builder: ::melior::ir::operation::OperationBuilder::new(#name, location),
                    #(#segment_fields,)*
                    _state: Default::default(),
                }
            }
        }
    }
}

pub fn generate_operation_builder_fn(builder: &TypeStateBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let arguments = builder.type_state().initial_arguments();

    quote! {
        /// Creates a builder.
        pub fn builder(
            context: &'c ::melior::Context,
            location: ::melior::ir::Location<'c>
        ) -> #identifier<'c, #(#arguments),*> {
            #identifier::new(context, location)
        }
    }
}

pub fn generate_default_constructor(builder: &TypeStateBuilder) -> TokenStream {
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
    let successor = builder.operation().successors().next().is_some();
    let unsafety = successor.then(|| quote!(unsafe));
    let safety_doc = successor.then(generate_successor_safety_doc);
    let body = quote! {
        #operation_identifier::builder(context, location)#(#builder_calls)*.build()
    };
    let body = if successor {
        quote! { unsafe { #body } }
    } else {
        body
    };

    quote! {
        #[allow(clippy::too_many_arguments)]
        #[doc = #doc]
        #safety_doc
        pub #unsafety fn #constructor_identifier<'c>(context: &'c ::melior::Context, #(#arguments),*) -> #operation_identifier<'c> {
            #body
        }
    }
}

fn generate_successor_safety_doc() -> TokenStream {
    quote! {
        #[doc = ""]
        #[doc = "# Safety"]
        #[doc = ""]
        #[doc = "The successor blocks must not be destroyed while the operation refers to them."]
    }
}
