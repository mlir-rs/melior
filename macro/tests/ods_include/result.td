include "mlir/IR/OpBase.td"

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
