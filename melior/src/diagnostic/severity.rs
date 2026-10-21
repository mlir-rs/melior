use crate::Error;
use mlir_sys::{
    MlirDiagnosticSeverity, MlirDiagnosticSeverity_MlirDiagnosticError,
    MlirDiagnosticSeverity_MlirDiagnosticNote, MlirDiagnosticSeverity_MlirDiagnosticRemark,
    MlirDiagnosticSeverity_MlirDiagnosticWarning,
};

/// Diagnostic severity.
#[derive(Clone, Copy, Debug)]
pub enum DiagnosticSeverity {
    Error,
    Note,
    Remark,
    Warning,
}

impl TryFrom<MlirDiagnosticSeverity> for DiagnosticSeverity {
    type Error = Error;

    fn try_from(severity: MlirDiagnosticSeverity) -> Result<Self, Error> {
        #[allow(non_upper_case_globals)]
        Ok(match severity {
            MlirDiagnosticSeverity_MlirDiagnosticError => Self::Error,
            MlirDiagnosticSeverity_MlirDiagnosticNote => Self::Note,
            MlirDiagnosticSeverity_MlirDiagnosticRemark => Self::Remark,
            MlirDiagnosticSeverity_MlirDiagnosticWarning => Self::Warning,
            _ => return Err(Error::UnknownDiagnosticSeverity(severity as _)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_error() {
        assert!(matches!(
            DiagnosticSeverity::try_from(MlirDiagnosticSeverity_MlirDiagnosticError),
            Ok(DiagnosticSeverity::Error)
        ));
    }

    #[test]
    fn convert_note() {
        assert!(matches!(
            DiagnosticSeverity::try_from(MlirDiagnosticSeverity_MlirDiagnosticNote),
            Ok(DiagnosticSeverity::Note)
        ));
    }

    #[test]
    fn convert_remark() {
        assert!(matches!(
            DiagnosticSeverity::try_from(MlirDiagnosticSeverity_MlirDiagnosticRemark),
            Ok(DiagnosticSeverity::Remark)
        ));
    }

    #[test]
    fn convert_warning() {
        assert!(matches!(
            DiagnosticSeverity::try_from(MlirDiagnosticSeverity_MlirDiagnosticWarning),
            Ok(DiagnosticSeverity::Warning)
        ));
    }

    #[test]
    fn convert_unknown() {
        assert_eq!(
            DiagnosticSeverity::try_from(42).unwrap_err(),
            Error::UnknownDiagnosticSeverity(42)
        );
    }
}
