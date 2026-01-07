use mlir_sys::{
    MlirOpPrintingFlags, mlirOpPrintingFlagsCreate, mlirOpPrintingFlagsDestroy,
    mlirOpPrintingFlagsElideLargeElementsAttrs, mlirOpPrintingFlagsEnableDebugInfo,
    mlirOpPrintingFlagsPrintGenericOpForm, mlirOpPrintingFlagsUseLocalScope,
};

/// Operation printing flags.
#[derive(Debug)]
pub struct OperationPrintingFlags(MlirOpPrintingFlags);

impl OperationPrintingFlags {
    /// Creates operation printing flags.
    pub fn new() -> Self {
        Self(unsafe { mlirOpPrintingFlagsCreate() })
    }

    /// Elides large elements attributes.
    pub fn elide_large_elements_attributes(self, limit: usize) -> Self {
        unsafe { mlirOpPrintingFlagsElideLargeElementsAttrs(self.0, limit as isize) }

        self
    }

    /// Enables debug info.
    pub fn enable_debug_info(self, enabled: bool, pretty_form: bool) -> Self {
        unsafe { mlirOpPrintingFlagsEnableDebugInfo(self.0, enabled, pretty_form) }

        self
    }

    /// Prints operations in a generic form.
    pub fn print_generic_operation_form(self) -> Self {
        unsafe { mlirOpPrintingFlagsPrintGenericOpForm(self.0) }

        self
    }

    /// Uses local scope.
    pub fn use_local_scope(self) -> Self {
        unsafe { mlirOpPrintingFlagsUseLocalScope(self.0) }

        self
    }

    /// Converts a printing flags into a raw object.
    pub const fn to_raw(&self) -> MlirOpPrintingFlags {
        self.0
    }
}

impl Drop for OperationPrintingFlags {
    fn drop(&mut self) {
        unsafe { mlirOpPrintingFlagsDestroy(self.0) }
    }
}

impl Default for OperationPrintingFlags {
    fn default() -> Self {
        Self::new()
    }
}
