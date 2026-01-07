//! Passes and pass managers.

pub mod r#async;
pub mod conversion;
pub mod external;
pub mod gpu;
pub mod linalg;
mod manager;
mod operation_manager;
pub mod sparse_tensor;
pub mod transform;

pub use self::{
    external::{ExternalPass, RunExternalPass, create_external},
    manager::PassManager,
    operation_manager::OperationPassManager,
};
use mlir_sys::MlirPass;

/// A pass.
pub struct Pass {
    raw: MlirPass,
}

impl Pass {
    /// Creates a pass from a raw function.
    ///
    /// # Safety
    ///
    /// A raw function must be valid.
    pub unsafe fn from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        Self {
            raw: unsafe { create_raw() },
        }
    }

    /// Creates a pass from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub const unsafe fn from_raw(raw: MlirPass) -> Self {
        Self { raw }
    }

    /// Converts a pass into a raw object.
    pub const fn to_raw(&self) -> MlirPass {
        self.raw
    }

    #[doc(hidden)]
    pub unsafe fn __private_from_raw_fn(create_raw: unsafe extern "C" fn() -> MlirPass) -> Self {
        unsafe { Self::from_raw_fn(create_raw) }
    }
}
