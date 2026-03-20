include "mlir/IR/OpBase.td"

def FirstAttrTest_Dialect : Dialect {
    let name = "first_attr_test";
    let cppNamespace = "::mlir::first_attr_test";
}

class FirstAttrTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<FirstAttrTest_Dialect, mnemonic, traits>;

// Mimics a constant-like op: result type derived from a TypeAttr attribute.
// The generated setter calls .value() on the TypeAttribute to get the result type.
def FirstAttrTest_ConstOp : FirstAttrTest_Op<"const", [FirstAttrDerivedResultType]> {
    let arguments = (ins TypeAttr:$result_type_attr);
    let results = (outs AnyType:$result);
}

// Non-TypeAttr case: result type derived from the attribute's own type.
// IntegerAttr's type is the integer type it holds (e.g. i32 for a 42 : i32 attr).
def FirstAttrTest_FromAttrOp : FirstAttrTest_Op<"from_attr", [FirstAttrDerivedResultType]> {
    let arguments = (ins AnyAttr:$value_attr);
    let results = (outs AnyType:$result);
}
