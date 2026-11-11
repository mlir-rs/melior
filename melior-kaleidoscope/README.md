# Melior Kaleidoscope

A compiler of the Toy language from [the MLIR Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) written in Rust with [Melior](https://github.com/mlir-rs/melior).

## Usage

```sh
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --emit mlir
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --emit mlir-affine --optimize
cargo run -p melior-kaleidoscope -- melior-kaleidoscope/examples/codegen.toy --emit jit --optimize
```

The `--emit` option selects an output like the tutorial's `-emit` option: `ast`, `mlir`, `mlir-affine`, `mlir-llvm`, `llvm`, or `jit`. The `--optimize` flag enables optimizations. An input file with the `.mlir` extension is read as an MLIR module instead of a Toy program.

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

## Differences from the tutorial

The Toy dialect is defined in TableGen as in the tutorial, and the `melior::dialect!` macro generates builders and accessors for its operations. However, the MLIR C API cannot register dialects, so the operations are built as ones of an unregistered dialect. As a result:

- The operations are printed in the generic format, and the custom assembly formats of the tutorial are not available.
- The operations have no verifiers, traits, or interfaces. Transformations relying on them in the tutorial, such as inlining, dead code elimination, common sub-expression elimination of constants, and shape inference, are implemented in Rust.
- The conversion framework is not available in the C API. Lowering is implemented as rewrite patterns applied by the greedy rewrite driver, followed by a check that no Toy operation is left.

The struct type of chapter 7 is represented by the builtin tuple type because custom types cannot be defined through the C API either. The `-emit=llvm` output is the plain translation of the module in the LLVM dialect. The target triple and data layout setup and the LLVM optimization pipeline of the tutorial are not applied because the MLIR C API does not expose them.
