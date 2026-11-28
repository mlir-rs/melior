use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseI8ArrayGet, mlirDenseI8ArrayGetElement,
};

dense_array_attribute!(
    DenseI8ArrayAttribute,
    i8,
    mlirDenseI8ArrayGet,
    mlirDenseI8ArrayGetElement,
    is_dense_i8_array
);

#[cfg(test)]
mod tests {
    use crate::test::create_test_context;

    use super::*;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseI8ArrayAttribute::new(&context, &[i8::MIN, -1, i8::MAX]);

        assert_eq!(attribute.element(0).unwrap(), i8::MIN);
        assert_eq!(attribute.element(1).unwrap(), -1);
        assert_eq!(attribute.element(2).unwrap(), i8::MAX);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseI8ArrayAttribute::new(&context, &[i8::MIN, -1, i8::MAX]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseI8ArrayAttribute::new(&context, &[i8::MIN, -1, i8::MAX]).into();

        assert!(DenseI8ArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseI8ArrayAttribute::try_from(attribute).is_err());
    }
}
