// A field whose setter appends elements to an operation. Its setter must be
// called before the ones of the later fields of the same kind.
#[derive(Debug)]
pub struct OrderedField {
    name: String,
    optional: bool,
}

impl OrderedField {
    pub const fn new(name: String, optional: bool) -> Self {
        Self { name, optional }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn is_optional(&self) -> bool {
        self.optional
    }
}
