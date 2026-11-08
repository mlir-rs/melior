// A field whose setter appends elements to an operation. Its setter must be
// called before the ones of the later fields of the same kind.
#[derive(Debug)]
pub struct OrderedField {
    optional: bool,
}

impl OrderedField {
    pub const fn new(optional: bool) -> Self {
        Self { optional }
    }

    pub const fn is_optional(&self) -> bool {
        self.optional
    }
}
