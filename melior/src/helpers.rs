//! Traits that extends the [`Block`](crate::ir::Block) type to aid in code
//! generation and consistency.

mod arith;
mod builtin;
mod llvm;

pub use arith::ArithBlockExt;
pub use builtin::BuiltinBlockExt;
pub use llvm::{GepIndex, LlvmBlockExt};
