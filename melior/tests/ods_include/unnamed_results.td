include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Test_Dialect : Dialect {
    let name = "unnamed_results";
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

// One unnamed fixed result.
def Test_SingleOp : Test_Op<"single"> {
    let arguments = (ins I32:$input);
    let results = (outs I32);
}

// A named result followed by an unnamed one.
def Test_MixedOp : Test_Op<"mixed"> {
    let results = (outs I32:$first, I64);
}

// A named variadic result followed by an unnamed fixed result.
def Test_VariadicOp : Test_Op<"variadic"> {
    let results = (outs Variadic<I32>:$values, I64);
}

// An unnamed result with a supported type inference.
def Test_InferOp : Test_Op<"infer", [SameOperandsAndResultType]> {
    let arguments = (ins I32:$input);
    let results = (outs I32);
}
