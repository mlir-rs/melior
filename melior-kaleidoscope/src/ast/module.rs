use super::{Record, dumper::Dumper};
use core::fmt::{self, Display, Formatter};

/// A module.
#[derive(Debug, PartialEq)]
pub struct Module {
    file_name: String,
    records: Vec<Record>,
}

impl Module {
    /// Creates a module.
    pub const fn new(file_name: String, records: Vec<Record>) -> Self {
        Self { file_name, records }
    }

    /// Returns a file name.
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Returns records.
    pub fn records(&self) -> &[Record] {
        &self.records
    }
}

impl Display for Module {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Dumper::new(formatter, &self.file_name).dump_module(self)
    }
}
