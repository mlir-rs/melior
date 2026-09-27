use crate::dialect::operation::{
    Attribute, Operation, OperationBuilder, OperationElement, OperationField, TypeInference,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Type;

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
        "A builder for {}.{}",
        builder.operation().documentation_name(),
        if builder.operation().segment_arrays().next().is_none() {
            ""
        } else {
            // Group sizes are recorded per declared group while the elements
            // themselves are appended in the order the setters are called. The
            // type parameters order the setters of required fields, but
            // optional fields have no type parameters, so their setters are
            // callable at any point of a builder chain.
            "\n\nThe sizes of the operand or result groups of this operation are \
            recorded in an `operandSegmentSizes` or `resultSegmentSizes` attribute. \
            Setters of optional fields must be called in the order the fields are \
            declared, as they are not ordered by the type parameters of a builder."
        }
    );
    let type_parameters = builder.type_state().parameters().collect::<Vec<_>>();
    let segment_fields = builder.operation().segment_arrays().map(|(kind, length)| {
        let identifier = kind.field_identifier();

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
            let identifier = kind.field_identifier();

            match field.segment() {
                Some((field_kind, index)) if field_kind == kind => {
                    let size = generate_segment_size(field);

                    quote! {
                        #identifier: {
                            let mut sizes = self.#identifier;
                            sizes[#index] = #size;
                            sizes
                        }
                    }
                }
                _ => quote! { #identifier: self.#identifier },
            }
        })
        .collect()
}

// Generates an in-place update of the field's group length, for setters
// mutating a builder instead of consuming it.
fn generate_segment_size_assignment(
    operation: &Operation,
    field: &impl OperationField,
) -> TokenStream {
    match field.segment() {
        Some((kind, index))
            if operation
                .segment_arrays()
                .any(|(array_kind, _)| array_kind == kind) =>
        {
            let identifier = kind.field_identifier();
            let size = generate_segment_size(field);

            quote! { self.#identifier[#index] = #size; }
        }
        _ => quote! {},
    }
}

// A group's length is the length of its setter's argument if that is a slice,
// which is the case for variadic groups, and 1 otherwise.
fn generate_segment_size(field: &impl OperationField) -> TokenStream {
    let identifier = field.singular_identifier();

    if matches!(field.parameter_type(), Type::Reference(_)) {
        quote! { #identifier.len() as i32 }
    } else {
        quote! { 1 }
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
        let segment_size_assignment = generate_segment_size_assignment(builder.operation(), field);

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    #segment_size_assignment
                    self.builder = self.builder.#add_identifier(#add_arguments);
                    self
                }
            }
        }
    } else {
        let parameters = builder.type_state().parameters_without(field.name());
        let arguments_set = builder.type_state().arguments_with(field.name(), true);
        let arguments_unset = builder.type_state().arguments_with(field.name(), false);
        let segment_size_fields = generate_segment_size_fields(builder.operation(), field);

        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    #builder_identifier {
                        context: self.context,
                        #(#segment_size_fields,)*
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
        let segment_size_assignment = generate_segment_size_assignment(builder.operation(), field);
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#parameters),*> {
                pub fn #identifier(mut self, #argument) -> #builder_identifier<'c, #(#parameters),*> {
                    let result_type = #type_access;
                    #segment_size_assignment
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
        let segment_size_fields = generate_segment_size_fields(builder.operation(), field);
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    let result_type = #type_access;
                    #builder_identifier {
                        context: self.context,
                        #(#segment_size_fields,)*
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
        let segment_size_fields = generate_segment_size_fields(builder.operation(), field);
        quote! {
            impl<'c, #(#parameters),*> #builder_identifier<'c, #(#arguments_unset),*> {
                pub fn #identifier(self, #argument) -> #builder_identifier<'c, #(#arguments_set),*> {
                    let result_type = #type_access;
                    #builder_identifier {
                        context: self.context,
                        #(#segment_size_fields,)*
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
    let add_segment_size_attributes = builder.operation().segment_arrays().map(|(kind, _)| {
        let field_identifier = kind.field_identifier();
        let name = kind.attribute_name();

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
        impl<'c> #identifier<'c, #(#arguments),*> {
            pub fn build(self) -> #operation_identifier<'c> {
                self.builder #(#add_segment_size_attributes)* #maybe_infer
                    .build().expect("valid operation").try_into().expect(#error)
            }
        }
    }
}

fn generate_new_fn(builder: &OperationBuilder) -> TokenStream {
    let identifier = builder.identifier();
    let name = &builder.operation().full_operation_name();
    let arguments = builder.type_state().arguments_with_all(false);
    let segment_fields = builder.operation().segment_arrays().map(|(kind, length)| {
        let identifier = kind.field_identifier();

        quote! { #identifier: [0; #length] }
    });

    quote! {
        impl<'c> #identifier<'c, #(#arguments),*> {
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
