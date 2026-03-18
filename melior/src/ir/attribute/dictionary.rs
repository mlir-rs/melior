use super::{Attribute, AttributeLike};
use crate::{Context, Error, StringRef, ir::Identifier};
use mlir_sys::{
    MlirAttribute, mlirDictionaryAttrGet, mlirDictionaryAttrGetElement,
    mlirDictionaryAttrGetElementByName, mlirDictionaryAttrGetNumElements, mlirNamedAttributeGet,
};

/// A dictionary attribute.
#[derive(Clone, Copy, Hash)]
pub struct DictionaryAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DictionaryAttribute<'c> {
    /// Creates a dictionary attribute.
    pub fn new(context: &'c Context, elements: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
        let named: Vec<_> = elements
            .iter()
            .map(|(id, attr)| unsafe { mlirNamedAttributeGet(id.to_raw(), attr.to_raw()) })
            .collect();
        unsafe {
            Self::from_raw(mlirDictionaryAttrGet(
                context.to_raw(),
                named.len() as isize,
                named.as_ptr(),
            ))
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        (unsafe { mlirDictionaryAttrGetNumElements(self.to_raw()) }) as usize
    }

    /// Checks if the dictionary attribute is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the element at the given index.
    pub fn element(&self, index: usize) -> Result<(Identifier<'c>, Attribute<'c>), Error> {
        if index < self.len() {
            let named = unsafe { mlirDictionaryAttrGetElement(self.to_raw(), index as isize) };
            Ok(unsafe {
                (
                    Identifier::from_raw(named.name),
                    Attribute::from_raw(named.attribute),
                )
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "dictionary element",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Returns the attribute with the given name, or `None` if not found.
    pub fn element_by_name(&self, name: &str) -> Option<Attribute<'c>> {
        unsafe {
            Attribute::from_option_raw(mlirDictionaryAttrGetElementByName(
                self.to_raw(),
                StringRef::new(name).to_raw(),
            ))
        }
    }
}

attribute_traits!(DictionaryAttribute, is_dictionary, "dictionary");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{attribute::IntegerAttribute, r#type::IntegerType},
        test::create_test_context,
    };

    #[test]
    fn new_empty() {
        let context = create_test_context();
        let attribute = DictionaryAttribute::new(&context, &[]);

        assert!(attribute.is_empty());
        assert_eq!(attribute.len(), 0);
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let id = Identifier::new(&context, "foo");
        let val = IntegerAttribute::new(IntegerType::new(&context, 64).into(), 42).into();
        let attribute = DictionaryAttribute::new(&context, &[(id, val)]);

        assert_eq!(attribute.len(), 1);
        assert!(!attribute.is_empty());
    }

    #[test]
    fn element() {
        let context = create_test_context();
        let id = Identifier::new(&context, "bar");
        let val = IntegerAttribute::new(IntegerType::new(&context, 64).into(), 7).into();
        let attribute = DictionaryAttribute::new(&context, &[(id, val)]);

        let (got_id, got_val) = attribute.element(0).unwrap();
        assert_eq!(got_id.as_string_ref().as_str().unwrap(), "bar");
        assert_eq!(got_val, val);
        assert!(matches!(
            attribute.element(1),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn element_by_name() {
        let context = create_test_context();
        let id = Identifier::new(&context, "baz");
        let val = IntegerAttribute::new(IntegerType::new(&context, 64).into(), 99).into();
        let attribute = DictionaryAttribute::new(&context, &[(id, val)]);

        assert_eq!(attribute.element_by_name("baz"), Some(val));
        assert_eq!(attribute.element_by_name("missing"), None);
    }
}
