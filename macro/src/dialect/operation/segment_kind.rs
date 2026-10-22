use crate::dialect::utility::segment_size_attribute_name;
use quote::format_ident;
use syn::Ident;

// A kind of elements whose group sizes are recorded in a segment size
// attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentKind {
    Operand,
    Result,
}

impl SegmentKind {
    const fn singular_name(self) -> &'static str {
        match self {
            Self::Operand => "operand",
            Self::Result => "result",
        }
    }

    pub fn field_identifier(self) -> Ident {
        format_ident!("{}_segment_sizes", self.singular_name())
    }

    pub fn attribute_name(self) -> String {
        segment_size_attribute_name(self.singular_name())
    }
}
