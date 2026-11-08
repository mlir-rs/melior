//! Diagnostics.

mod handler_id;
mod severity;

pub use self::{handler_id::DiagnosticHandlerId, severity::DiagnosticSeverity};
use crate::{Error, ir::Location, utility::print_callback};
use core::{
    ffi::c_void,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
};
use mlir_sys::{
    MlirDiagnostic, mlirDiagnosticGetLocation, mlirDiagnosticGetNote, mlirDiagnosticGetNumNotes,
    mlirDiagnosticGetSeverity, mlirDiagnosticPrint,
};

/// A diagnostic.
#[derive(Debug)]
pub struct Diagnostic<'c> {
    raw: MlirDiagnostic,
    phantom: PhantomData<&'c ()>,
}

impl<'c> Diagnostic<'c> {
    /// Returns a location.
    pub fn location(&self) -> Location<'c> {
        unsafe { Location::from_raw(mlirDiagnosticGetLocation(self.raw)) }
    }

    /// Returns a severity.
    pub fn severity(&self) -> DiagnosticSeverity {
        DiagnosticSeverity::try_from(unsafe { mlirDiagnosticGetSeverity(self.raw) })
            .unwrap_or_else(|error| unreachable!("{}", error))
    }

    /// Returns a number of notes.
    pub fn note_count(&self) -> usize {
        (unsafe { mlirDiagnosticGetNumNotes(self.raw) }) as usize
    }

    /// Returns a note at the given index.
    pub fn note(&self, index: usize) -> Result<Self, Error> {
        if index < self.note_count() {
            Ok(unsafe { Self::from_raw(mlirDiagnosticGetNote(self.raw, index as isize)) })
        } else {
            Err(Error::PositionOutOfBounds {
                name: "diagnostic note",
                value: self.to_string(),
                index,
            })
        }
    }

    /// Creates a diagnostic from a raw object.
    ///
    /// # Safety
    ///
    /// A raw object must be valid.
    pub unsafe fn from_raw(raw: MlirDiagnostic) -> Self {
        Self {
            raw,
            phantom: Default::default(),
        }
    }
}

impl Display for Diagnostic<'_> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        let mut data = (formatter, Ok(()));

        unsafe {
            mlirDiagnosticPrint(
                self.raw,
                Some(print_callback),
                &mut data as *mut _ as *mut c_void,
            );
        }

        data.1
    }
}

#[cfg(test)]
mod tests {
    use crate::{Context, ir::Module};
    use alloc::rc::Rc;
    use core::cell::RefCell;

    #[test]
    fn handle_diagnostic() {
        let message = Rc::new(RefCell::new(None));
        let context = Context::new();

        context.attach_diagnostic_handler({
            let message = message.clone();

            move |diagnostic| {
                *message.borrow_mut() = Some(diagnostic.to_string());
                true
            }
        });

        Module::parse(&context, "foo");

        assert_eq!(
            message.borrow().as_deref(),
            Some("custom op 'foo' is unknown (tried 'builtin.foo' as well)")
        );
    }
}
