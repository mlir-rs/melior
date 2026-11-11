use crate::{
    canonicalize::canonicalize, error::Error, inliner::inline_functions,
    lower_to_affine::lower_to_affine, lower_to_llvm::lower_to_llvm, shape_inference::infer_shapes,
};
use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    ir::Module,
    pass::{PassManager, affine, transform},
    utility::{register_all_dialects, register_all_llvm_translations},
};

/// A compilation target.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Action {
    /// The Toy dialect.
    Mlir,
    /// The affine dialect after partial lowering.
    MlirAffine,
    /// The LLVM dialect.
    MlirLlvm,
    /// LLVM IR.
    Llvm,
    /// JIT execution.
    Jit,
}

/// Creates an MLIR context for the Toy compiler.
///
/// The Toy dialect is not registered in the context and its operations are
/// built as ones of an unregistered dialect.
pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);
    register_all_llvm_translations(&context);

    context
}

/// Transforms a module in the Toy dialect for a compilation target.
pub fn process<'c>(
    context: &'c Context,
    module: &mut Module<'c>,
    action: Action,
    optimize: bool,
) -> Result<(), Error> {
    let lowering_to_affine = action >= Action::MlirAffine;
    let lowering_to_llvm = action >= Action::MlirLlvm;

    if optimize || lowering_to_affine {
        // Inline all functions into the main one so that shapes of every
        // operation can be inferred.
        inline_functions(context, module)?;
        canonicalize(context, module)?;
        infer_shapes(module)?;
        canonicalize(context, module)?;
    }

    if lowering_to_affine {
        lower_to_affine(context, module)?;

        let pass_manager = PassManager::new(context);
        let function_manager = pass_manager.nested_under("func.func");

        function_manager.add_pass(transform::create_canonicalizer_pass());
        function_manager.add_pass(transform::create_cse_pass());

        if optimize {
            function_manager.add_pass(affine::create_affine_loop_fusion());
            function_manager.add_pass(affine::create_affine_scalar_replacement());
        }

        pass_manager.run(module)?;
    }

    if lowering_to_llvm {
        lower_to_llvm(context, module)?;
    }

    Ok(())
}

/// Runs the main function of a module in the LLVM dialect with a JIT compiler.
pub fn run_jit(module: &Module, optimize: bool) -> Result<(), Error> {
    let engine = ExecutionEngine::new(module, if optimize { 3 } else { 0 }, &[], false, false)?;

    // SAFETY: The main function takes no arguments and returns no values.
    unsafe { engine.invoke_packed("main", &mut []) }?;

    Ok(())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{mlir_gen::generate, parse::parse};

    pub fn compile<'c>(
        context: &'c Context,
        source: &str,
        action: Action,
        optimize: bool,
    ) -> Result<Module<'c>, Error> {
        let mut module = generate(context, &parse(source, "test.toy")?)?;

        process(context, &mut module, action, optimize)?;

        Ok(module)
    }

    // The JIT compiler cannot resolve the `printf` function on Windows.
    #[cfg(not(windows))]
    mod jit {
        use super::*;

        #[test]
        fn print() {
            let context = create_context();
            let module = compile(
                &context,
                include_str!("../examples/jit.toy"),
                Action::Jit,
                false,
            )
            .unwrap();

            run_jit(&module, false).unwrap();
        }

        #[test]
        fn optimized_multiply_transpose() {
            let context = create_context();
            let module = compile(
                &context,
                include_str!("../examples/codegen.toy"),
                Action::Jit,
                true,
            )
            .unwrap();

            run_jit(&module, true).unwrap();
        }
    }
}
