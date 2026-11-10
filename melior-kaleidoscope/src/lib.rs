//! The Toy language compiler from [the MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) built with Melior.

pub mod ast;
mod canonicalize;
mod compile;
mod error;
mod inliner;
mod location;
mod lower_to_affine;
mod lower_to_llvm;
mod mlir_gen;
pub mod parse;
mod shape_inference;
mod utility;

pub use self::{
    canonicalize::canonicalize,
    compile::{Action, create_context, process, run_jit},
    error::Error,
    inliner::inline_functions,
    location::Location,
    lower_to_affine::lower_to_affine,
    lower_to_llvm::lower_to_llvm,
    mlir_gen::generate,
    shape_inference::infer_shapes,
};

melior::dialect! {
    name: "toy",
    files: ["src/toy.td"],
    include_directory_env_vars: ["CARGO_MANIFEST_DIR"],
}
