use super::Record;

/// A module.
#[derive(Debug, PartialEq)]
pub struct Module {
    path: String,
    records: Vec<Record>,
}

impl Module {
    /// Creates a module.
    pub const fn new(path: String, records: Vec<Record>) -> Self {
        Self { path, records }
    }

    /// Returns a path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns records.
    pub fn records(&self) -> &[Record] {
        &self.records
    }
}
