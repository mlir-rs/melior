use super::OperationPassManager;
use crate::{
    context::Context,
    ir::{operation::OperationPrintingFlags, Module},
    logical_result::LogicalResult,
    pass::Pass,
    string_ref::StringRef,
    Error,
};
use mlir_sys::{
    mlirPassManagerAddOwnedPass, mlirPassManagerCreate, mlirPassManagerDestroy,
    mlirPassManagerEnableIRPrinting, mlirPassManagerEnableVerifier,
    mlirPassManagerGetAsOpPassManager, mlirPassManagerGetNestedUnder, mlirPassManagerRunOnOp,
    MlirPassManager,
};
use std::{marker::PhantomData, mem::forget, path::PathBuf};

/// A pass manager.
pub struct PassManager<'c> {
    raw: MlirPassManager,
    _context: PhantomData<&'c Context>,
}

impl<'c> PassManager<'c> {
    /// Creates a pass manager.
    pub fn new(context: &Context) -> Self {
        Self {
            raw: unsafe { mlirPassManagerCreate(context.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Returns an operation pass manager for nested operations corresponding to
    /// a given name.
    pub fn nested_under(&self, name: &str) -> OperationPassManager<'c, '_> {
        let name = StringRef::new(name);

        unsafe {
            OperationPassManager::from_raw(mlirPassManagerGetNestedUnder(self.raw, name.to_raw()))
        }
    }

    /// Adds a pass.
    pub fn add_pass(&self, pass: Pass) {
        unsafe { mlirPassManagerAddOwnedPass(self.raw, pass.to_raw()) }
    }

    /// Enables a verifier.
    pub fn enable_verifier(&self, enabled: bool) {
        unsafe { mlirPassManagerEnableVerifier(self.raw, enabled) }
    }

    /// Enable IR printing.
    /// The treePrintingPath argument is an optional path to a directory
    /// where the dumps will be produced. If it isn't provided then dumps
    /// are produced to stderr.
    pub fn enable_ir_printing(&self, options: &PassIrPrintingOptions) {
        unsafe {
            mlirPassManagerEnableIRPrinting(
                self.raw,
                options.before_all,
                options.after_all,
                options.module_scope,
                options.on_change,
                options.on_failure,
                options.flags.to_raw(),
                StringRef::new(&options.tree_printing_path.display().to_string()).to_raw(),
            )
        }
    }

    /// Runs passes added to a pass manager against a module.
    pub fn run(&self, module: &mut Module) -> Result<(), Error> {
        let result = LogicalResult::from_raw(unsafe {
            mlirPassManagerRunOnOp(self.raw, module.as_operation().to_raw())
        });

        if result.is_success() {
            Ok(())
        } else {
            Err(Error::RunPass)
        }
    }

    /// Converts a pass manager to an operation pass manager.
    pub fn as_operation_pass_manager(&self) -> OperationPassManager<'c, '_> {
        unsafe { OperationPassManager::from_raw(mlirPassManagerGetAsOpPassManager(self.raw)) }
    }

    /// Creates a PassManager from the given raw pointer.
    ///
    /// # Safety
    /// Caller must ensure this is a valid PassManager pointer.
    pub unsafe fn from_raw(raw: MlirPassManager) -> Self {
        Self {
            raw,
            _context: Default::default(),
        }
    }

    /// Gets the raw object of this pass manager.
    pub const fn to_raw(&self) -> MlirPassManager {
        self.raw
    }

    /// Converts a PassManager into an owned raw object.
    pub const fn into_raw(self) -> MlirPassManager {
        let raw = self.raw;
        forget(self);
        raw
    }
}

impl Drop for PassManager<'_> {
    fn drop(&mut self) {
        unsafe { mlirPassManagerDestroy(self.raw) }
    }
}

#[derive(Debug)]
pub struct PassIrPrintingOptions {
    pub before_all: bool,
    pub after_all: bool,
    pub module_scope: bool,
    pub on_change: bool,
    pub on_failure: bool,
    pub flags: OperationPrintingFlags,
    pub tree_printing_path: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{Location, Module},
        pass::{self, transform::register_print_op_stats},
        test::create_test_context,
        utility::parse_pass_pipeline,
    };
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[test]
    fn new() {
        let context = create_test_context();

        PassManager::new(&context);
    }

    #[test]
    fn add_pass() {
        let context = create_test_context();

        PassManager::new(&context).add_pass(pass::conversion::create_func_to_llvm());
    }

    #[test]
    fn enable_verifier() {
        let context = create_test_context();

        PassManager::new(&context).enable_verifier(true);
    }

    // TODO Enable this test.
    // #[test]
    // fn enable_ir_printing() {
    //     let context = Context::new();

    //     PassManager::new(&context).enable_ir_printing();
    // }

    #[test]
    fn run() {
        let context = create_test_context();
        let manager = PassManager::new(&context);

        manager.add_pass(pass::conversion::create_func_to_llvm());
        manager
            .run(&mut Module::new(Location::unknown(&context)))
            .unwrap();
    }

    #[test]
    fn run_on_function() {
        let context = create_test_context();

        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
                "
            ),
        )
        .unwrap();

        let manager = PassManager::new(&context);
        manager.add_pass(pass::transform::create_print_op_stats());

        assert_eq!(manager.run(&mut module), Ok(()));
    }

    #[test]
    fn run_on_function_in_nested_module() {
        let context = create_test_context();

        let mut module = Module::parse(
            &context,
            indoc!(
                "
                func.func @foo(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }

                module {
                    func.func @bar(%arg0 : f32) -> f32 {
                        %res = arith.addf %arg0, %arg0 : f32
                        return %res : f32
                    }
                }
                "
            ),
        )
        .unwrap();

        let manager = PassManager::new(&context);
        manager
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());

        assert_eq!(manager.run(&mut module), Ok(()));

        let manager = PassManager::new(&context);
        manager
            .nested_under("builtin.module")
            .nested_under("func.func")
            .add_pass(pass::transform::create_print_op_stats());

        assert_eq!(manager.run(&mut module), Ok(()));
    }

    #[test]
    fn print_pass_pipeline() {
        let context = create_test_context();
        let manager = PassManager::new(&context);
        let function_manager = manager.nested_under("func.func");

        function_manager.add_pass(pass::transform::create_print_op_stats());

        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "any(func.func(print-op-stats{json=false}))"
        );
        assert_eq!(
            function_manager.to_string(),
            "func.func(print-op-stats{json=false})"
        );
    }

    #[test]
    fn parse_pass_pipeline_() {
        let context = Context::new();
        let manager = PassManager::new(&context);

        insta::assert_snapshot!(parse_pass_pipeline(
            manager.as_operation_pass_manager(),
            "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
        )
        .unwrap_err());

        register_print_op_stats();

        assert_eq!(
            parse_pass_pipeline(
                manager.as_operation_pass_manager(),
                "builtin.module(func.func(print-op-stats{json=false}),\
                func.func(print-op-stats{json=false}))"
            ),
            Ok(())
        );

        assert_eq!(
            manager.as_operation_pass_manager().to_string(),
            "builtin.module(func.func(print-op-stats{json=false}),\
            func.func(print-op-stats{json=false}))"
        );
    }
}
