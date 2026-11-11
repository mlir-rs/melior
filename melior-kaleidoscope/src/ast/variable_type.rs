/// A variable type.
#[derive(Debug, Eq, PartialEq)]
pub enum VariableType {
    /// A struct type of a name.
    Struct(String),
    /// A tensor type of a shape, which is unknown if empty.
    Tensor(Vec<u64>),
}
