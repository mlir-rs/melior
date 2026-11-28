use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseF64ArrayGet, mlirDenseF64ArrayGetElement,
};

dense_array_attribute!(
    DenseF64ArrayAttribute,
    f64,
    mlirDenseF64ArrayGet,
    mlirDenseF64ArrayGetElement,
    is_dense_f64_array
);

#[cfg(test)]
mod tests {
    use crate::test::create_test_context;

    use super::*;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseF64ArrayAttribute::new(&context, &[-1.5, 0.25, f64::MAX]);

        assert_eq!(attribute.element(0).unwrap(), -1.5);
        assert_eq!(attribute.element(1).unwrap(), 0.25);
        assert_eq!(attribute.element(2).unwrap(), f64::MAX);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseF64ArrayAttribute::new(&context, &[-1.5, 0.25, f64::MAX]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute =
            DenseF64ArrayAttribute::new(&context, &[-1.5, 0.25, f64::MAX]).into();

        assert!(DenseF64ArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseF64ArrayAttribute::try_from(attribute).is_err());
    }
}
