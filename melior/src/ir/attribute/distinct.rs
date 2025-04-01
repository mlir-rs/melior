use super::Attribute;
use mlir_sys::{mlirDisctinctAttrCreate, MlirAttribute};

/// A disctinct attribute.
#[derive(Clone, Copy)]
pub struct DisctinctAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> DisctinctAttribute<'c> {
    /// Creates a disctinct attribute.
    pub fn new(referenced_attr: &Attribute<'c>) -> Self {
        unsafe { Self::from_raw(mlirDisctinctAttrCreate(referenced_attr.raw)) }
    }
}

attribute_traits_no_try_from!(DisctinctAttribute);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::attribute::BoolAttribute, test::create_test_context};

    #[test]
    fn value() {
        let context = create_test_context();
        let bool_attr = BoolAttribute::new(&context, true);
        let value = DisctinctAttribute::new(&bool_attr.into());
        let _value: Attribute = value.into();
    }
}
