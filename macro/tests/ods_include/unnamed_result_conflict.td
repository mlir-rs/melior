include "mlir/IR/OpBase.td"

def ResultConflict_Dialect : Dialect {
    let name = "result_conflict";
}

def ResultConflict_FooOp : Op<ResultConflict_Dialect, "foo"> {
    let results = (outs I32:$result_1, I32);
}

def RegionConflict_Dialect : Dialect {
    let name = "region_conflict";
}

def RegionConflict_FooOp : Op<RegionConflict_Dialect, "foo"> {
    let results = (outs I32);
    let regions = (region AnyRegion:$result_0);
}
