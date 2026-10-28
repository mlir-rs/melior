#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariadicKind {
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
    ) -> Result<Option<Self>, &'static str> {
        Ok(if unfixed_count <= 1 {
            None
        } else if same_size {
            Some(Self::SameSize)
        } else if attribute_sized {
            Some(Self::AttributeSized)
        } else {
            // TODO: Support multiple variadic operands/results without these traits.
            return Err(
                "multiple variadic operands/results require SameVariadicOperandSize, \
                SameVariadicResultSize, or AttrSizedOperandSegments/AttrSizedResultSegments trait",
            );
        })
    }
}
