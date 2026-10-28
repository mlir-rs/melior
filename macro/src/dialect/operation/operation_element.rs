use super::{ElementPosition, OperationField};

pub trait OperationElement: OperationField {
    fn is_variadic(&self) -> bool;
    fn position(&self) -> ElementPosition;

    fn is_unfixed(&self) -> bool {
        self.is_variadic() || self.is_optional()
    }
}
