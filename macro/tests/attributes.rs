mod utility;

use melior::{
    Context,
    ir::{
        Attribute, Identifier, Location, Type,
        attribute::{
            ArrayAttribute, BoolAttribute, DenseElementsAttribute, DenseI32ArrayAttribute,
            DenseI64ArrayAttribute, DictionaryAttribute, FlatSymbolRefAttribute, FloatAttribute,
            IntegerAttribute, StridedLayoutAttribute, StringAttribute, TypeAttribute,
        },
        r#type::{IntegerType, RankedTensorType},
    },
};
use utility::*;

melior_macro::dialect! {
    name: "attribute_test",
    files: ["macro/tests/ods_include/attributes.td"],
}

fn context() -> Context {
    let context = create_test_context();
    context.set_allow_unregistered_dialects(true);
    context
}

#[test]
fn array_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let element = BoolAttribute::new(&context, true).into();
    let attribute = ArrayAttribute::new(&context, &[element]);

    let operation = attribute_test::array(&context, attribute, location);

    let value: ArrayAttribute = operation.value().unwrap();
    assert_eq!(value.len(), 1);
    assert_eq!(value.element(0).unwrap(), element);
}

#[test]
fn any_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute: Attribute = BoolAttribute::new(&context, true).into();

    let operation = attribute_test::any(&context, attribute, location);

    let value: Attribute = operation.value().unwrap();
    assert_eq!(value, attribute);
}

#[test]
fn bool_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = BoolAttribute::new(&context, true);

    let operation = attribute_test::flag(&context, attribute, location);

    let value: BoolAttribute = operation.value().unwrap();
    assert!(value.value());
}

#[test]
fn dense_elements_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let tensor = RankedTensorType::new(&[2], Type::float64(&context), None).into();
    let attribute = DenseElementsAttribute::f64_values(tensor, &[1.5, 2.5]);

    let operation = attribute_test::dense_elements(&context, attribute, location);

    let value: DenseElementsAttribute = operation.value().unwrap();
    assert_eq!(value.len(), 2);
    assert_eq!(value.f64_element(1).unwrap(), 2.5);
}

#[test]
fn dense_i32_array_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);

    let operation = attribute_test::dense_i_32_array(&context, attribute, location);

    let value: DenseI32ArrayAttribute = operation.value().unwrap();
    assert_eq!(value.len(), 3);
    assert_eq!(value.element(2).unwrap(), 3);
}

#[test]
fn dense_i64_array_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = DenseI64ArrayAttribute::new(&context, &[1, 2, 3]);

    let operation = attribute_test::dense_i_64_array(&context, attribute, location);

    let value: DenseI64ArrayAttribute = operation.value().unwrap();
    assert_eq!(value.len(), 3);
    assert_eq!(value.element(2).unwrap(), 3);
}

#[test]
fn dictionary_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let entry = BoolAttribute::new(&context, true).into();
    let attribute =
        DictionaryAttribute::new(&context, &[(Identifier::new(&context, "key"), entry)]);

    let operation = attribute_test::dictionary(&context, attribute, location);

    let value: DictionaryAttribute = operation.value().unwrap();
    assert_eq!(value.len(), 1);
    assert_eq!(value.element_by_name("key"), Some(entry));
}

#[test]
fn flat_symbol_ref_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = FlatSymbolRefAttribute::new(&context, "callee");

    let operation = attribute_test::flat_symbol_ref(&context, attribute, location);

    let value: FlatSymbolRefAttribute = operation.value().unwrap();
    assert_eq!(value.value(), "callee");
}

#[test]
fn float_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = FloatAttribute::new(&context, Type::float64(&context), 1.5);

    let operation = attribute_test::float(&context, attribute, location);

    let value: FloatAttribute = operation.value().unwrap();
    assert_eq!(value.value(), 1.5);
}

#[test]
fn integer_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let i64 = IntegerType::new(&context, 64).into();
    let attribute = IntegerAttribute::new(i64, 42);

    let operation = attribute_test::integer(&context, attribute, location);

    let value: IntegerAttribute = operation.value().unwrap();
    assert_eq!(value.value(), 42);
}

#[test]
fn string_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = StringAttribute::new(&context, "text");

    let operation = attribute_test::string(&context, attribute, location);

    let value: StringAttribute = operation.value().unwrap();
    assert_eq!(value.value(), "text");
}

#[test]
fn strided_layout_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let attribute = StridedLayoutAttribute::new(&context, 4, &[8, 1]);

    let operation = attribute_test::strided_layout(&context, attribute, location);

    let value: StridedLayoutAttribute = operation.value().unwrap();
    assert_eq!(value.offset(), 4);
    assert_eq!(value.stride_count(), 2);
    assert_eq!(value.stride(0).unwrap(), 8);
}

#[test]
fn type_attr() {
    let context = context();
    let location = Location::unknown(&context);
    let r#type = Type::float64(&context);
    let attribute = TypeAttribute::new(r#type);

    let operation = attribute_test::r#type(&context, attribute, location);

    let value: TypeAttribute = operation.value().unwrap();
    assert_eq!(value.value(), r#type);
}
