include "mlir/IR/OpBase.td"

def RegionTest_Dialect : Dialect {
    let name = "region_test";
    let cppNamespace = "::mlir::region_test";
}

class RegionTest_Op<string mnemonic, list<Trait> traits = []> :
        Op<RegionTest_Dialect, mnemonic, traits>;

def RegionTest_SingleOp : RegionTest_Op<"single"> {
    let regions = (region SizedRegion<1>:$defaultRegion);
}

def RegionTest_VariadicOp : RegionTest_Op<"variadic"> {
    let regions = (region SizedRegion<1>:$defaultRegion,
                   VariadicRegion<SizedRegion<1>>:$otherRegions);
}
