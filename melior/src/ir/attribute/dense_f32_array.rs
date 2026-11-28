use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseF32ArrayGet, mlirDenseF32ArrayGetElement,
};

dense_array_attribute!(
    DenseF32ArrayAttribute,
    f32,
    mlirDenseF32ArrayGet,
    mlirDenseF32ArrayGetElement,
    is_dense_f32_array
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseF32ArrayAttribute::new(&context, &[-1.5, 0.25, f32::MAX]);

        assert_eq!(attribute.element(0).unwrap(), -1.5);
        assert_eq!(attribute.element(1).unwrap(), 0.25);
        assert_eq!(attribute.element(2).unwrap(), f32::MAX);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseF32ArrayAttribute::new(&context, &[-1.5, 0.25, f32::MAX]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseF32ArrayAttribute::new(&context, &[-1.5, 0.25, f32::MAX]).into();

        assert!(DenseF32ArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseF32ArrayAttribute::try_from(attribute).is_err());
    }
}
