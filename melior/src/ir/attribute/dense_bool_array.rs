use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseBoolArrayGet,
    mlirDenseBoolArrayGetElement,
};

dense_array_attribute!(
    DenseBoolArrayAttribute,
    bool,
    mlirDenseBoolArrayGet,
    mlirDenseBoolArrayGetElement,
    is_dense_bool_array
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseBoolArrayAttribute::new(&context, &[true, false, true]);

        assert!(attribute.element(0).unwrap());
        assert!(!attribute.element(1).unwrap());
        assert!(attribute.element(2).unwrap());
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseBoolArrayAttribute::new(&context, &[true, false, true]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseBoolArrayAttribute::new(&context, &[true, false, true]).into();

        assert!(DenseBoolArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseBoolArrayAttribute::try_from(attribute).is_err());
    }
}
