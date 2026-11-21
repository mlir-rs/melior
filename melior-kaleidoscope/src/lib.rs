//! The Toy language compiler from [the MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/).

pub mod ast;
mod canonicalize;
mod compile;
mod error;
mod generate;
mod inliner;
mod location;
mod lower_to_affine;
mod lower_to_llvm;
mod parse;
mod shape_inference;
#[cfg(test)]
mod test;
mod utility;

pub use self::{
    canonicalize::canonicalize,
    compile::{Action, compile, create_context, run_jit},
    error::Error,
    generate::generate,
    inliner::inline_functions,
    location::Location,
    lower_to_affine::lower_to_affine,
    lower_to_llvm::lower_to_llvm,
    parse::parse,
    shape_inference::infer_shapes,
};

melior::dialect! {
    name: "toy",
    files: ["src/toy.td"],
    include_directory_env_vars: ["CARGO_MANIFEST_DIR"],
}
