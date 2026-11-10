include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def SameTypeTest_Dialect : Dialect {
    let name = "same_type_test";
    let cppNamespace = "::mlir::same_type_test";
}

class SameTypeTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<SameTypeTest_Dialect, mnemonic, traits>;

// Mimics a comb-style binary op: result type == operand type, inferred from operands.
def SameTypeTest_AddOp : SameTypeTest_Op<"add", [SameOperandsAndResultType]> {
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger:$result);
}

// Variadic-first-operand case: exercises the &operands[0] indexing path in
// generate_same_operands_first_fn. A non-empty slice works; an empty slice panics.
def SameTypeTest_SumOp : SameTypeTest_Op<"sum", [SameOperandsAndResultType]> {
    let arguments = (ins Variadic<AnyInteger>:$inputs);
    let results = (outs AnyInteger:$result);
}

// Non-integer type: exercises the same code path with a tensor type.
def SameTypeTest_TensorOp : SameTypeTest_Op<"tensor_add", [SameOperandsAndResultType]> {
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}
