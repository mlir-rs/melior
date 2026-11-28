use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseI16ArrayGet, mlirDenseI16ArrayGetElement,
};

dense_array_attribute!(
    DenseI16ArrayAttribute,
    i16,
    mlirDenseI16ArrayGet,
    mlirDenseI16ArrayGetElement,
    is_dense_i16_array
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseI16ArrayAttribute::new(&context, &[i16::MIN, -1, i16::MAX]);

        assert_eq!(attribute.element(0).unwrap(), i16::MIN);
        assert_eq!(attribute.element(1).unwrap(), -1);
        assert_eq!(attribute.element(2).unwrap(), i16::MAX);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseI16ArrayAttribute::new(&context, &[i16::MIN, -1, i16::MAX]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseI16ArrayAttribute::new(&context, &[i16::MIN, -1, i16::MAX]).into();

        assert!(DenseI16ArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseI16ArrayAttribute::try_from(attribute).is_err());
    }
}
