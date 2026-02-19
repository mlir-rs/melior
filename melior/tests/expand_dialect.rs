#![cfg(feature = "ods-dialects")]

use melior::dialect;

dialect! {
    name: "affine",
    files: ["IR/AffineOps.td", "TransformOps/AffineTransformOps.td", "IR/AffineMemoryOpInterfaces.td"],
    include_directories: ["mlir/Dialect/Affine"],
}

dialect! {
    name: "arith",
    files: ["mlir/Dialect/Arith/IR/ArithOps.td"],
    include_directories: [env!("FOO")],
}
