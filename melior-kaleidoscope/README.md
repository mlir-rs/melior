# Melior Kaleidoscope

A compiler of the Toy language from [the MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) written in Rust with [Melior](https://github.com/mlir-rs/melior).

## Usage

```sh
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --emit mlir
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --emit mlir-affine --optimize
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --optimize
```

The `--emit` option selects an output like the tutorial's `-emit` option: `ast`, `mlir`, `mlir-affine`, `mlir-llvm`, or `llvm`. The main function is JIT-compiled and executed when it is omitted. The `--optimize` flag enables optimizations. An input file with the `.mlir` extension is read as an MLIR module instead of a Toy program.

## Tutorial chapters

| Chapter                                                      | Implementation                                |
| ------------------------------------------------------------ | --------------------------------------------- |
| 1. Toy language and AST                                      | `src/parse.rs` and `src/ast.rs`               |
| 2. Emitting basic MLIR                                       | `src/toy.td` and `src/mlir_gen.rs`            |
| 3. High-level language-specific analysis and transformation  | `src/canonicalize.rs`                         |
| 4. Enabling generic transformation with interfaces           | `src/inliner.rs` and `src/shape_inference.rs` |
| 5. Partial lowering to lower-level dialects for optimization | `src/lower_to_affine.rs`                      |
| 6. Lowering to LLVM and code generation                      | `src/lower_to_llvm.rs` and `src/compile.rs`   |
| 7. Adding a composite type to Toy                            | Struct types in the modules above             |
