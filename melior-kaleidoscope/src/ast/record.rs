use super::{Function, Struct};

/// A top-level record.
#[derive(Debug, PartialEq)]
pub enum Record {
    /// A function definition.
    Function(Function),
    /// A struct definition.
    Struct(Struct),
}
