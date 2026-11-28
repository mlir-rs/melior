macro_rules! attribute_traits {
    ($name: ident, $is_type: ident, $string: expr) => {
        impl<'c> $name<'c> {
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attribute: unsafe { Attribute::from_raw(raw) },
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

        impl<'c> core::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> core::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Display::fmt(self, formatter)
            }
        }
    };
}

macro_rules! attribute_traits_without_try_from {
    ($name: ident) => {
        impl<'c> $name<'c> {
            #[allow(unused)]
            unsafe fn from_raw(raw: MlirAttribute) -> Self {
                Self {
                    attribute: unsafe { Attribute::from_raw(raw) },
                }
            }
        }

        impl<'c> crate::ir::attribute::AttributeLike<'c> for $name<'c> {
            fn to_raw(&self) -> mlir_sys::MlirAttribute {
                self.attribute.to_raw()
            }
        }

        impl<'c> core::fmt::Display for $name<'c> {
            fn fmt(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Display::fmt(&self.attribute, formatter)
            }
        }

        impl<'c> core::fmt::Debug for $name<'c> {
            fn fmt(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Display::fmt(self, formatter)
            }
        }
    };
}

macro_rules! dense_array_attribute {
    ($name:ident, $type:ident, $create:ident, $get_element:ident, $is_type:ident) => {
        #[doc = concat!(" A dense ", stringify!($type), " array attribute.")]
        #[derive(Clone, Copy, Hash)]
        pub struct $name<'c> {
            attribute: Attribute<'c>,
        }

        impl<'c> $name<'c> {
            #[doc = concat!(" Creates a dense ", stringify!($type), " array attribute.")]
            pub fn new(context: &'c Context, values: &[$type]) -> Self {
                unsafe {
                    Self::from_raw($create(
                        context.to_raw(),
                        values.len() as isize,
                        dense_array_attribute!(@values $type, values).as_ptr(),
                    ))
                }
            }

            /// Returns a length.
            pub fn len(&self) -> usize {
                (unsafe { mlirDenseArrayGetNumElements(self.attribute.to_raw()) }) as usize
            }

            /// Checks if an array is empty.
            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            /// Returns an element.
            pub fn element(&self, index: usize) -> Result<$type, Error> {
                if index < self.len() {
                    Ok(unsafe { $get_element(self.attribute.to_raw(), index as isize) })
                } else {
                    Err(Error::PositionOutOfBounds {
                        name: "array element",
                        value: self.to_string(),
                        index,
                    })
                }
            }
        }

        attribute_traits!(
            $name,
            $is_type,
            concat!("dense ", stringify!($type), " array")
        );
    };
    (@values bool, $values:ident) => {
        $values
            .iter()
            .copied()
            .map(core::ffi::c_int::from)
            .collect::<Vec<_>>()
    };
    (@values $type:ident, $values:ident) => {
        $values
    };
}
