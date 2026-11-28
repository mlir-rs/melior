use super::{Attribute, AttributeLike, FlatSymbolRefAttribute};
use crate::{Context, Error, StringRef};
use core::str::Utf8Error;
use mlir_sys::{
    MlirAttribute, mlirSymbolRefAttrGet, mlirSymbolRefAttrGetLeafReference,
    mlirSymbolRefAttrGetNestedReference, mlirSymbolRefAttrGetNumNestedReferences,
    mlirSymbolRefAttrGetRootReference,
};

/// A symbol ref attribute.
#[derive(Clone, Copy, Hash)]
pub struct SymbolRefAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> SymbolRefAttribute<'c> {
    /// Creates a symbol ref attribute.
    pub fn new(
        context: &'c Context,
        symbol: &str,
        nested_references: &[FlatSymbolRefAttribute<'c>],
    ) -> Self {
        unsafe {
            Self::from_raw(mlirSymbolRefAttrGet(
                context.to_raw(),
                StringRef::new(symbol).to_raw(),
                nested_references.len() as isize,
                nested_references.as_ptr() as *const _,
            ))
        }
    }

    /// Returns a root reference.
    pub fn root_reference(&self) -> Result<&'c str, Utf8Error> {
        unsafe { StringRef::from_raw(mlirSymbolRefAttrGetRootReference(self.to_raw())) }.as_str()
    }

    /// Returns a leaf reference.
    pub fn leaf_reference(&self) -> Result<&'c str, Utf8Error> {
        unsafe { StringRef::from_raw(mlirSymbolRefAttrGetLeafReference(self.to_raw())) }.as_str()
    }

    /// Returns a number of nested references.
    pub fn nested_reference_count(&self) -> usize {
        (unsafe { mlirSymbolRefAttrGetNumNestedReferences(self.to_raw()) }) as usize
    }

    /// Returns a nested reference at a position.
    pub fn nested_reference(&self, index: usize) -> Result<FlatSymbolRefAttribute<'c>, Error> {
        if index < self.nested_reference_count() {
            FlatSymbolRefAttribute::try_from(unsafe {
                Attribute::from_raw(mlirSymbolRefAttrGetNestedReference(
                    self.to_raw(),
                    index as isize,
                ))
            })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "nested reference",
                value: self.to_string(),
                index,
            })
        }
    }
}

attribute_traits!(SymbolRefAttribute, is_symbol_ref, "symbol ref");

from_subtypes!(SymbolRefAttribute, FlatSymbolRefAttribute);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn new() {
        let context = create_test_context();

        assert_eq!(
            SymbolRefAttribute::new(&context, "foo", &[]).to_string(),
            "@foo"
        );
    }

    #[test]
    fn new_with_nested_references() {
        let context = create_test_context();

        assert_eq!(
            SymbolRefAttribute::new(
                &context,
                "foo",
                &[
                    FlatSymbolRefAttribute::new(&context, "bar"),
                    FlatSymbolRefAttribute::new(&context, "baz"),
                ],
            )
            .to_string(),
            "@foo::@bar::@baz"
        );
    }

    #[test]
    fn root_reference() {
        let context = create_test_context();
        let attribute = SymbolRefAttribute::new(
            &context,
            "foo",
            &[FlatSymbolRefAttribute::new(&context, "bar")],
        );

        assert_eq!(attribute.root_reference(), Ok("foo"));
    }

    #[test]
    fn leaf_reference() {
        let context = create_test_context();
        let attribute = SymbolRefAttribute::new(
            &context,
            "foo",
            &[
                FlatSymbolRefAttribute::new(&context, "bar"),
                FlatSymbolRefAttribute::new(&context, "baz"),
            ],
        );

        assert_eq!(attribute.leaf_reference(), Ok("baz"));
    }

    #[test]
    fn leaf_reference_without_nested_references() {
        let context = create_test_context();
        let attribute = SymbolRefAttribute::new(&context, "foo", &[]);

        assert_eq!(attribute.leaf_reference(), Ok("foo"));
    }

    #[test]
    fn fail_to_get_non_utf8_root_reference() {
        let context = create_test_context();
        let attribute =
            SymbolRefAttribute::try_from(Attribute::parse(&context, r#"@"\FF"::@foo"#).unwrap())
                .unwrap();

        assert!(attribute.root_reference().is_err());
    }

    #[test]
    fn fail_to_get_non_utf8_leaf_reference() {
        let context = create_test_context();
        let attribute =
            SymbolRefAttribute::try_from(Attribute::parse(&context, r#"@foo::@"\FF""#).unwrap())
                .unwrap();

        assert!(attribute.leaf_reference().is_err());
    }

    #[test]
    fn nested_reference_count() {
        let context = create_test_context();

        assert_eq!(
            SymbolRefAttribute::new(&context, "foo", &[]).nested_reference_count(),
            0
        );
        assert_eq!(
            SymbolRefAttribute::new(
                &context,
                "foo",
                &[
                    FlatSymbolRefAttribute::new(&context, "bar"),
                    FlatSymbolRefAttribute::new(&context, "baz"),
                ],
            )
            .nested_reference_count(),
            2
        );
    }

    #[test]
    fn nested_reference() {
        let context = create_test_context();
        let attribute = SymbolRefAttribute::new(
            &context,
            "foo",
            &[
                FlatSymbolRefAttribute::new(&context, "bar"),
                FlatSymbolRefAttribute::new(&context, "baz"),
            ],
        );

        assert_eq!(attribute.nested_reference(0).unwrap().value(), Ok("bar"));
        assert_eq!(attribute.nested_reference(1).unwrap().value(), Ok("baz"));
        assert!(matches!(
            attribute.nested_reference(2),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn convert_flat_symbol_ref() {
        let context = create_test_context();

        assert_eq!(
            Attribute::from(SymbolRefAttribute::from(FlatSymbolRefAttribute::new(
                &context, "foo"
            ))),
            SymbolRefAttribute::new(&context, "foo", &[]).into()
        );
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute = SymbolRefAttribute::new(
            &context,
            "foo",
            &[FlatSymbolRefAttribute::new(&context, "bar")],
        )
        .into();

        assert!(SymbolRefAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_accepts_flat_symbol_ref() {
        let context = create_test_context();
        let attribute: Attribute = FlatSymbolRefAttribute::new(&context, "foo").into();

        assert!(SymbolRefAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(SymbolRefAttribute::try_from(attribute).is_err());
    }
}
