use quote::format_ident;
use syn::Ident;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldKind {
    Result,
    Operand,
    Region,
    Successor,
    Attribute,
}

impl FieldKind {
    pub const fn singular_name(self) -> &'static str {
        match self {
            Self::Result => "result",
            Self::Operand => "operand",
            Self::Region => "region",
            Self::Successor => "successor",
            Self::Attribute => "attribute",
        }
    }

    pub const fn plural_name(self) -> &'static str {
        match self {
            Self::Result => "results",
            Self::Operand => "operands",
            Self::Region => "regions",
            Self::Successor => "successors",
            Self::Attribute => "attributes",
        }
    }

    pub fn add_identifier(self) -> Ident {
        format_ident!("add_{}", self.plural_name())
    }

    pub fn segment_size_field_identifier(self) -> Ident {
        format_ident!("{}_segment_sizes", self.singular_name())
    }

    // The name of the attribute MLIR uses to store the sizes of the operand or
    // result groups of an operation with the `AttrSizedOperandSegments` or
    // `AttrSizedResultSegments` trait.
    pub fn segment_size_attribute_name(self) -> String {
        format!("{}SegmentSizes", self.singular_name())
    }
}
