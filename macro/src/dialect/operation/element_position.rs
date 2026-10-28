use super::VariadicKind;

// A position of an operand or result among the elements of the same kind in its
// operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ElementPosition {
    kind: Option<VariadicKind>,
    index: usize,
    preceding_unfixed_count: usize,
    unfixed_count: usize,
}

impl ElementPosition {
    pub const fn new(
        kind: Option<VariadicKind>,
        index: usize,
        preceding_unfixed_count: usize,
        unfixed_count: usize,
    ) -> Self {
        Self {
            kind,
            index,
            preceding_unfixed_count,
            unfixed_count,
        }
    }

    pub const fn kind(&self) -> Option<VariadicKind> {
        self.kind
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

    // The position of an element's group in its operation's segment size
    // attribute, if the operation sizes its groups by one.
    pub const fn segment_index(&self) -> Option<usize> {
        match self.kind {
            Some(VariadicKind::AttributeSized) => Some(self.index),
            _ => None,
        }
    }
}
