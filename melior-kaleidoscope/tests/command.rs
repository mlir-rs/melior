#![expect(missing_docs)]

use pretty_assertions::assert_eq;
use std::{
    path::Path,
    process::{Command, Output},
};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_melior-kaleidoscope"))
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples"))
        .args(arguments)
        .output()
        .unwrap()
}

fn run_successfully(arguments: &[&str]) -> String {
    let output = run(arguments);

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn emit_ast() {
    insta::assert_snapshot!(run_successfully(&["ast.toy", "--emit", "ast"]));
}

#[test]
fn emit_struct_ast() {
    insta::assert_snapshot!(run_successfully(&["struct_codegen.toy", "--emit", "ast"]));
}

#[test]
fn emit_mlir() {
    insta::assert_snapshot!(run_successfully(&["codegen.toy", "--emit", "mlir"]));
}

#[test]
fn emit_optimized_mlir() {
    insta::assert_snapshot!(run_successfully(&[
        "struct_codegen.toy",
        "--emit",
        "mlir",
        "--opt"
    ]));
}

#[test]
fn emit_optimized_mlir_of_transposes() {
    insta::assert_snapshot!(run_successfully(&[
        "transpose_transpose.toy",
        "--emit",
        "mlir",
        "--opt"
    ]));
}

#[test]
fn emit_optimized_mlir_of_reshapes() {
    insta::assert_snapshot!(run_successfully(&[
        "trivial_reshape.toy",
        "--emit",
        "mlir",
        "--opt"
    ]));
}

#[test]
fn emit_mlir_affine() {
    insta::assert_snapshot!(run_successfully(&["codegen.toy", "--emit", "mlir-affine"]));
}

#[test]
fn emit_optimized_mlir_affine() {
    insta::assert_snapshot!(run_successfully(&[
        "codegen.toy",
        "--emit",
        "mlir-affine",
        "--opt"
    ]));
}

#[test]
fn emit_optimized_mlir_llvm() {
    insta::assert_snapshot!(run_successfully(&[
        "codegen.toy",
        "--emit",
        "mlir-llvm",
        "--opt"
    ]));
}

#[test]
fn emit_mlir_affine_from_mlir() {
    insta::assert_snapshot!(run_successfully(&[
        "affine_lowering.mlir",
        "--emit",
        "mlir-affine",
        "--opt"
    ]));
}

// The JIT compiler cannot resolve the `printf` function on Windows.
#[cfg(not(windows))]
#[test]
fn run_jit() {
    assert_eq!(
        run_successfully(&["jit.toy", "--emit", "jit"]),
        "1.000000 2.000000 \n3.000000 4.000000 \n"
    );
}

#[cfg(not(windows))]
#[test]
fn run_optimized_jit() {
    assert_eq!(
        run_successfully(&["codegen.toy", "--emit", "jit", "--opt"]),
        "1.000000 16.000000 \n4.000000 25.000000 \n9.000000 36.000000 \n"
    );
}

#[cfg(not(windows))]
#[test]
fn run_jit_with_struct() {
    assert_eq!(
        run_successfully(&["struct_codegen.toy", "--emit", "jit"]),
        "1.000000 16.000000 \n4.000000 25.000000 \n9.000000 36.000000 \n"
    );
}

#[test]
fn fail_to_parse() {
    let output = run(&["invalid.toy", "--emit", "ast"]);

    assert!(!output.status.success());
    insta::assert_snapshot!(String::from_utf8(output.stderr).unwrap());
}

#[test]
fn fail_to_infer_shapes() {
    let output = run(&["shape_mismatch.toy", "--emit", "mlir", "--opt"]);

    assert!(!output.status.success());
    insta::assert_snapshot!(String::from_utf8(output.stderr).unwrap());
}
