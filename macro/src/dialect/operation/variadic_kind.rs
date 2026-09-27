#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariadicKind {
    Simple {
        unfixed_seen: bool,
    },
    SameSize {
        unfixed_count: usize,
        preceding_simple_count: usize,
        preceding_variadic_count: usize,
    },
    AttributeSized {
        segment_index: usize,
    },
    // TODO Support variadic-of-variadic operands.
    // https://mlir.llvm.org/docs/DefiningDialects/Operations/#variadicofvariadic-operands
}

impl VariadicKind {
    pub fn new(
        unfixed_count: usize,
        same_size: bool,
        attribute_sized: bool,
    ) -> Result<Self, &'static str> {
        if unfixed_count <= 1 {
            Ok(Self::Simple {
                unfixed_seen: false,
            })
        } else if same_size {
            Ok(Self::SameSize {
                unfixed_count,
                preceding_simple_count: 0,
                preceding_variadic_count: 0,
            })
        } else if attribute_sized {
            Ok(Self::AttributeSized { segment_index: 0 })
        } else {
            // TODO: Support multiple variadic operands/results without these traits.
            Err(
                "multiple variadic operands/results require SameVariadicOperandSize, \
                SameVariadicResultSize, or AttrSizedOperandSegments/AttrSizedResultSegments trait",
            )
        }
    }

    // The position of an element's group in its operation's segment size
    // attribute, if the operation sizes its groups by one.
    pub const fn segment_index(&self) -> Option<usize> {
        match self {
            Self::AttributeSized { segment_index } => Some(*segment_index),
            _ => None,
        }
    }
}
