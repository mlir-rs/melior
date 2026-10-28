#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariadicKind {
    SameSize,
    AttributeSized,
    // TODO Support variadic-of-variadic operands.
    // https://mlir.llvm.org/docs/DefiningDialects/Operations/#variadicofvariadic-operands
}
