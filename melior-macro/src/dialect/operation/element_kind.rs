use super::FieldKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElementKind {
    Operand,
    Result,
}

impl From<ElementKind> for FieldKind {
    fn from(kind: ElementKind) -> Self {
        match kind {
            ElementKind::Operand => Self::Operand,
            ElementKind::Result => Self::Result,
        }
    }
}
