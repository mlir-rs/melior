include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def AttributeTest_Dialect : Dialect {
    let name = "attribute_test";
    let cppNamespace = "::mlir::attribute_test";
}

class AttributeTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<AttributeTest_Dialect, mnemonic, traits>;

def AttributeTest_ArrayOp : AttributeTest_Op<"array"> {
    let arguments = (ins ArrayAttr:$value);
}

def AttributeTest_AnyOp : AttributeTest_Op<"any"> {
    let arguments = (ins AnyAttr:$value);
}

def AttributeTest_BoolOp : AttributeTest_Op<"flag"> {
    let arguments = (ins BoolAttr:$value);
}

def AttributeTest_DenseElementsOp : AttributeTest_Op<"dense_elements"> {
    let arguments = (ins F64ElementsAttr:$value);
}

def AttributeTest_DenseI32ArrayOp : AttributeTest_Op<"dense_i32_array"> {
    let arguments = (ins DenseI32ArrayAttr:$value);
}

def AttributeTest_DenseI64ArrayOp : AttributeTest_Op<"dense_i64_array"> {
    let arguments = (ins DenseI64ArrayAttr:$value);
}

def AttributeTest_DictionaryOp : AttributeTest_Op<"dictionary"> {
    let arguments = (ins DictionaryAttr:$value);
}

def AttributeTest_FlatSymbolRefOp : AttributeTest_Op<"flat_symbol_ref"> {
    let arguments = (ins FlatSymbolRefAttr:$value);
}

def AttributeTest_FloatOp : AttributeTest_Op<"float"> {
    let arguments = (ins F64Attr:$value);
}

def AttributeTest_IntegerOp : AttributeTest_Op<"integer"> {
    let arguments = (ins I64Attr:$value);
}

def AttributeTest_StringOp : AttributeTest_Op<"string"> {
    let arguments = (ins StrAttr:$value);
}

def AttributeTest_StridedLayoutOp : AttributeTest_Op<"strided_layout"> {
    let arguments = (ins StridedLayoutAttr:$value);
}

def AttributeTest_TypeOp : AttributeTest_Op<"type"> {
    let arguments = (ins TypeAttr:$value);
}