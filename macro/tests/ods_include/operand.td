include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def OperandTest_Dialect : Dialect {
    let name = "operand_test";
    let cppNamespace = "::mlir::operand_test";
}

class OperandTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<OperandTest_Dialect, mnemonic, traits>;

def OperandTest_SimpleOp : OperandTest_Op<"simple"> {
    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32:$res);
}

def OperandTest_VariadicOp : OperandTest_Op<"variadic"> {
    let arguments = (ins I32:$first, Variadic<I32>:$others);
    let results = (outs I32:$res);
}

def OperandTest_AttributeSizedOp :
        OperandTest_Op<"attribute_sized", [AttrSizedOperandSegments]> {
    let arguments = (ins Variadic<I32>:$sizes, Variadic<I32>:$others);
    let results = (outs I32:$res);
}
