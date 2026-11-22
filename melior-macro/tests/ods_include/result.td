include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def ResultTest_Dialect : Dialect {
    let name = "result_test";
    let cppNamespace = "::mlir::result_test";
}

class ResultTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<ResultTest_Dialect, mnemonic, traits>;

def ResultTest_AttributeSizedOp :
        ResultTest_Op<"attribute_sized", [AttrSizedResultSegments]> {
    let results = (outs
        AnyType:$first,
        Optional<AnyType>:$optional,
        Variadic<AnyType>:$others
    );
}

def ResultTest_SameSizeOp :
        ResultTest_Op<"same_size", [SameVariadicResultSize]> {
    let results = (outs Variadic<AnyType>:$lhs, Variadic<AnyType>:$rhs);
}

def ResultTest_InferenceInterfaceOp :
        ResultTest_Op<"inference_interface", [InferTypeOpInterface]> {
    let results = (outs AnyType:$output);
}
