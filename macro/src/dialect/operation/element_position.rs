use super::VariadicKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementPosition {
    index: usize,
    preceding_unfixed_count: usize,
    unfixed_count: usize,
    variadic_kind: Option<VariadicKind>,
}

impl ElementPosition {
    pub const fn new(
        index: usize,
        preceding_unfixed_count: usize,
        unfixed_count: usize,
        variadic_kind: Option<VariadicKind>,
    ) -> Self {
        Self {
            index,
            preceding_unfixed_count,
            unfixed_count,
            variadic_kind,
        }
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub const fn preceding_unfixed_count(&self) -> usize {
        self.preceding_unfixed_count
    }

    pub const fn unfixed_count(&self) -> usize {
        self.unfixed_count
    }

    pub const fn variadic_kind(&self) -> Option<VariadicKind> {
        self.variadic_kind
    }
}
