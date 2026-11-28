use super::{Attribute, AttributeLike};
use crate::{Context, Error};
use mlir_sys::{
    MlirAttribute, mlirDenseArrayGetNumElements, mlirDenseI32ArrayGet, mlirDenseI32ArrayGetElement,
};

dense_array_attribute!(
    DenseI32ArrayAttribute,
    i32,
    mlirDenseI32ArrayGet,
    mlirDenseI32ArrayGetElement,
    is_dense_i32_array
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::create_test_context;

    #[test]
    fn element() {
        let context = create_test_context();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.element(0).unwrap(), 1);
        assert_eq!(attribute.element(1).unwrap(), 2);
        assert_eq!(attribute.element(2).unwrap(), 3);
        assert!(matches!(
            attribute.element(3),
            Err(Error::PositionOutOfBounds { .. })
        ));
    }

    #[test]
    fn len() {
        let context = create_test_context();
        let attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]);

        assert_eq!(attribute.len(), 3);
    }

    #[test]
    fn try_from_accepts_its_own_kind() {
        let context = create_test_context();
        let attribute: Attribute = DenseI32ArrayAttribute::new(&context, &[1, 2, 3]).into();

        assert!(DenseI32ArrayAttribute::try_from(attribute).is_ok());
    }

    #[test]
    fn try_from_rejects_another_kind() {
        let context = create_test_context();
        let attribute = Attribute::unit(&context);

        assert!(DenseI32ArrayAttribute::try_from(attribute).is_err());
    }
}
