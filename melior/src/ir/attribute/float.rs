use super::{Attribute, AttributeLike};
use crate::{
    ir::{Type, TypeLike},
    Context, Error,
};
use mlir_sys::{mlirFloatAttrDoubleGet, MlirAttribute};
use std::fmt::{self, Debug, Display, Formatter};

/// A float attribute.
#[derive(Clone, Copy)]
pub struct FloatAttribute<'c> {
    attribute: Attribute<'c>,
}

impl<'c> FloatAttribute<'c> {
    /// Creates a float attribute.
    pub fn new(context: &'c Context, number: f64, r#type: Type<'c>) -> Self {
        unsafe {
            Self::from_raw(mlirFloatAttrDoubleGet(
                context.to_raw(),
                r#type.to_raw(),
                number,
            ))
        }
    }

    unsafe fn from_raw(raw: MlirAttribute) -> Self {
        Self {
            attribute: Attribute::from_raw(raw),
        }
    }
}

impl<'c> AttributeLike<'c> for FloatAttribute<'c> {
    fn to_raw(&self) -> MlirAttribute {
        self.attribute.to_raw()
    }
}

impl<'c> TryFrom<Attribute<'c>> for FloatAttribute<'c> {
    type Error = Error;

    fn try_from(attribute: Attribute<'c>) -> Result<Self, Self::Error> {
        if attribute.is_float() {
            Ok(unsafe { Self::from_raw(attribute.to_raw()) })
        } else {
            Err(Error::AttributeExpected("float", format!("{}", attribute)))
        }
    }
}

impl<'c> Display for FloatAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(&self.attribute, formatter)
    }
}

impl<'c> Debug for FloatAttribute<'c> {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        Display::fmt(self, formatter)
    }
}
