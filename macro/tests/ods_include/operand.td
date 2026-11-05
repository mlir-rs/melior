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

def OperandTest_VariadicBeforeSingleOp : OperandTest_Op<"variadic_before_single"> {
    let arguments = (ins Variadic<I32>:$others, I32:$last);
    let results = (outs I32:$res);
}

def OperandTest_OptionalBeforeSingleOp : OperandTest_Op<"optional_before_single"> {
    let arguments = (ins Optional<I32>:$optional, I32:$last);
    let results = (outs I32:$res);
}

def OperandTest_AttributeSizedOp :
        OperandTest_Op<"attribute_sized", [AttrSizedOperandSegments]> {
    let arguments = (ins Variadic<I32>:$sizes, Variadic<I32>:$others);
    let results = (outs I32:$res);
}

def OperandTest_AttributeSizedOptionalOp :
        OperandTest_Op<"attribute_sized_optional", [AttrSizedOperandSegments]> {
    let arguments = (ins I32:$first, Optional<I32>:$optional, Variadic<I32>:$others);
    let results = (outs I32:$res);
}

def OperandTest_SameSizeOp :
        OperandTest_Op<"same_size", [SameVariadicOperandSize]> {
    let arguments = (ins Variadic<I32>:$lhs, Variadic<I32>:$rhs);
    let results = (outs I32:$res);
}

def OperandTest_SameSizeWithSinglesOp :
        OperandTest_Op<"same_size_with_singles", [SameVariadicOperandSize]> {
    let arguments = (ins
        I32:$first,
        Variadic<I32>:$lhs,
        I32:$middle,
        Variadic<I32>:$rhs,
        I32:$last
    );
    let results = (outs I32:$res);
}
