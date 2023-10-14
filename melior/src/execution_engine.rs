use crate::{ir::Module, logical_result::LogicalResult, string_ref::StringRef, Context, Error};
use mlir_sys::{
    mlirExecutionEngineCreate, mlirExecutionEngineDestroy, mlirExecutionEngineDumpToObjectFile,
    mlirExecutionEngineInvokePacked, mlirExecutionEngineRegisterSymbol, MlirExecutionEngine,
};

/// An execution engine.
pub struct ExecutionEngine {
    raw: MlirExecutionEngine,
}

impl ExecutionEngine {
    /// Creates an execution engine.
    pub fn new<'c>(
        context: &'c Context,
        module: &Module<'c>,
        optimization_level: usize,
        shared_library_paths: &[&str],
        enable_object_dump: bool,
    ) -> Self {
        Self {
            raw: unsafe {
                mlirExecutionEngineCreate(
                    module.to_raw(),
                    optimization_level as i32,
                    shared_library_paths.len() as i32,
                    shared_library_paths
                        .iter()
                        .map(|&string| StringRef::from_str(context, string).to_raw())
                        .collect::<Vec<_>>()
                        .as_ptr(),
                    enable_object_dump,
                )
            },
        }
    }

    /// Invokes a function in a module. The `arguments` argument includes
    /// pointers to results of the function as well as arguments.
    ///
    /// # Safety
    ///
    /// This function modifies memory locations pointed by the `arguments`
    /// argument. If those pointers are invalid or misaligned, calling this
    /// function might result in undefined behavior.
    pub unsafe fn invoke_packed(
        &self,
        context: &Context,
        name: &str,
        arguments: &mut [*mut ()],
    ) -> Result<(), Error> {
        let result = LogicalResult::from_raw(mlirExecutionEngineInvokePacked(
            self.raw,
            StringRef::from_str(context, name).to_raw(),
            arguments.as_mut_ptr() as _,
        ));

        if result.is_success() {
            Ok(())
        } else {
            Err(Error::InvokeFunction)
        }
    }

    /// Register a symbol. This symbol will be accessible to the JIT'd codes.
    ///
    /// # Safety
    ///
    /// This function makes a pointer accessible to the execution engine. If a
    /// given pointer is invalid or misaligned, calling this function might
    /// result in undefined behavior.
    pub unsafe fn register_symbol(&self, context: &Context, name: &str, ptr: *mut ()) {
        mlirExecutionEngineRegisterSymbol(
            self.raw,
            StringRef::from_str(context, name).to_raw(),
            ptr as _,
        );
    }

    /// Dumps a module to an object file.
    pub fn dump_to_object_file(&self, context: &Context, path: &str) {
        unsafe {
            mlirExecutionEngineDumpToObjectFile(
                self.raw,
                StringRef::from_str(context, path).to_raw(),
            )
        }
    }
}

impl Drop for ExecutionEngine {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pass, test::create_test_context};

    #[test]
    fn invoke_packed() {
        let context = create_test_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());

        pass_manager
            .nested_under(&context, "func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        let engine = ExecutionEngine::new(&context, &module, 2, &[], false);

        let mut argument = 42;
        let mut result = -1;

        assert_eq!(
            unsafe {
                engine.invoke_packed(
                    &context,
                    "add",
                    &mut [
                        &mut argument as *mut i32 as *mut (),
                        &mut result as *mut i32 as *mut (),
                    ],
                )
            },
            Ok(())
        );

        assert_eq!(argument, 42);
        assert_eq!(result, 84);
    }

    #[test]
    fn dump_to_object_file() {
        let context = create_test_context();

        let mut module = Module::parse(
            &context,
            r#"
            module {
                func.func @add(%arg0 : i32) -> i32 {
                    %res = arith.addi %arg0, %arg0 : i32
                    return %res : i32
                }
            }
            "#,
        )
        .unwrap();

        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_func_to_llvm());

        pass_manager
            .nested_under(&context, "func.func")
            .add_pass(pass::conversion::create_arith_to_llvm());

        assert_eq!(pass_manager.run(&mut module), Ok(()));

        ExecutionEngine::new(&context, &module, 2, &[], true)
            .dump_to_object_file(&context, "/tmp/melior/test.o");
    }
}
