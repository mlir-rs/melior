macro_rules! type_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirType) -> Self {
                Self {
                    r#type: unsafe { Type::from_raw(raw) },
                }
            }
        }

        impl<'c> TryFrom<crate::ir::r#type::Type<'c>> for $name<'c> {
            type Error = crate::Error;

            fn try_from(r#type: crate::ir::r#type::Type<'c>) -> Result<Self, Self::Error> {
                if r#type.$is_type() {
                    Ok(unsafe { Self::from_raw(r#type.to_raw()) })
                } else {
                    Err(Error::TypeExpected($string, r#type.to_string()))
                }
            }
        }

        impl<'c> crate::ir::r#type::TypeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirType {
                self.r#type.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.r#type, formatter)
            }
        }
    };
}
