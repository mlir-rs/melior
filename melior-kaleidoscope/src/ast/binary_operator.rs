use core::fmt::{self, Display, Formatter};

/// A binary operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOperator {
    /// A struct member access.
    Access,
    /// An addition.
    Add,
    /// A multiplication.
    Multiply,
    /// A subtraction.
    Subtract,
}

impl Display for BinaryOperator {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(
            formatter,
            "{}",
            match self {
                Self::Access => '.',
                Self::Add => '+',
                Self::Multiply => '*',
                Self::Subtract => '-',
            }
        )
    }
}
