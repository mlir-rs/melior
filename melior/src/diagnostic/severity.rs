use crate::Error;
use mlir_sys::{
    MlirDiagnosticSeverity_MlirDiagnosticError, MlirDiagnosticSeverity_MlirDiagnosticNote,
    MlirDiagnosticSeverity_MlirDiagnosticRemark, MlirDiagnosticSeverity_MlirDiagnosticWarning,
};

/// Diagnostic severity.
#[derive(Clone, Copy, Debug)]
pub enum DiagnosticSeverity {
    Error,
    Note,
    Remark,
    Warning,
}

impl TryFrom<u32> for DiagnosticSeverity {
    type Error = Error;

    fn try_from(severity: u32) -> Result<Self, Error> {
        #[allow(non_upper_case_globals)]
        Ok(match severity {
            x if x == MlirDiagnosticSeverity_MlirDiagnosticError as u32 => Self::Error,
            x if x == MlirDiagnosticSeverity_MlirDiagnosticNote as u32 => Self::Note,
            x if x == MlirDiagnosticSeverity_MlirDiagnosticRemark as u32 => Self::Remark,
            x if x == MlirDiagnosticSeverity_MlirDiagnosticWarning as u32 => Self::Warning,
            _ => return Err(Error::UnknownDiagnosticSeverity(severity)),
        })
    }
}

impl TryFrom<i32> for DiagnosticSeverity {
    type Error = Error;

    fn try_from(severity: i32) -> Result<Self, Error> {
        Self::try_from(severity as u32)
    }
}
