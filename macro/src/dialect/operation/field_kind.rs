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

    pub const fn type_parameter_prefix(self) -> &'static str {
        match self {
            Self::Result => "T",
            Self::Operand => "O",
            Self::Region => "R",
            Self::Successor => "S",
            Self::Attribute => "A",
        }
    }

    // The name of the attribute MLIR uses to store the sizes of the operand or
    // result groups of an operation with the `AttrSizedOperandSegments` or
    // `AttrSizedResultSegments` trait.
    pub fn segment_size_attribute_name(self) -> String {
        format!("{}SegmentSizes", self.singular_name())
    }
}
