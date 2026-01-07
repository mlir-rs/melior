use crate::{
    Context,
    ir::{
        Attribute, Identifier,
        attribute::{IntegerAttribute, TypeAttribute},
    },
};

const ATTRIBUTE_COUNT: usize = 3;

// spell-checker: disable

/// alloca options.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllocaOptions<'c> {
    align: Option<IntegerAttribute<'c>>,
    elem_type: Option<TypeAttribute<'c>>,
    inalloca: bool,
}

impl<'c> AllocaOptions<'c> {
    /// Creates load/store options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the alignment.
    pub fn align(mut self, align: Option<IntegerAttribute<'c>>) -> Self {
        self.align = align;
        self
    }

    /// Sets the elem_type, not needed if the returned pointer is not opaque.
    pub fn elem_type(mut self, elem_type: Option<TypeAttribute<'c>>) -> Self {
        self.elem_type = elem_type;
        self
    }

    /// Sets the inalloca flag.
    pub fn inalloca(mut self, inalloca: bool) -> Self {
        self.inalloca = inalloca;
        self
    }

    pub(super) fn into_attributes(
        self,
        context: &'c Context,
    ) -> Vec<(Identifier<'c>, Attribute<'c>)> {
        let mut attributes = Vec::with_capacity(ATTRIBUTE_COUNT);

        if let Some(align) = self.align {
            attributes.push((Identifier::new(context, "alignment"), align.into()));
        }

        if let Some(elem_type) = self.elem_type {
            attributes.push((Identifier::new(context, "elem_type"), elem_type.into()));
        }

        if self.inalloca {
            attributes.push((
                Identifier::new(context, "inalloca"),
                Attribute::unit(context),
            ));
        }

        attributes
    }
}
