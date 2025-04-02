macro_rules! attribute_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attribute: Attribute::from_raw(raw),
                }
            }
        }

        impl<'c> TryFrom<crate::ir::attribute::Attribute<'c>> for $name<'c> {
            type Error = crate::Error;

            fn try_from(
                attribute: crate::ir::attribute::Attribute<'c>,
            ) -> Result<Self, Self::Error> {
                if attribute.$is_type() {
                    Ok(unsafe { Self::from_raw(attribute.to_raw()) })
                } else {
                    Err(Error::AttributeExpected($string, attribute.to_string()))
                }
            }
        }

        impl<'c> crate::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, formatter)
            }
        }
    };
}

macro_rules! attribute_traits_no_try_from {
    ($name: ident) => {
        impl<'c> $name<'c> {
            #[allow(unused)]
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attribute: Attribute::from_raw(raw),
                }
            }
        }

        impl<'c> crate::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> std::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> std::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, formatter)
            }
        }
    };
}
