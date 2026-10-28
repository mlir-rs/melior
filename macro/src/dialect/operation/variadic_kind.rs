#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariadicKind {
    Simple,
    SameSize,
    AttributeSized,
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
            Ok(Self::Simple)
        } else if same_size {
            Ok(Self::SameSize)
        } else if attribute_sized {
            Ok(Self::AttributeSized)
        } else {
            // TODO: Support multiple variadic operands/results without these traits.
            Err(
                "multiple variadic operands/results require SameVariadicOperandSize, \
                SameVariadicResultSize, or AttrSizedOperandSegments/AttrSizedResultSegments trait",
            )
        }
    }
}
