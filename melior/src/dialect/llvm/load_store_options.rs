use crate::{
    Context,
    ir::{
        Attribute, Identifier,
        attribute::{ArrayAttribute, IntegerAttribute},
    },
};

const ATTRIBUTE_COUNT: usize = 7;

// spell-checker: disable

/// Load/store options.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoadStoreOptions<'c> {
    align: Option<IntegerAttribute<'c>>,
    volatile: bool,
    nontemporal: bool,
    access_groups: Option<ArrayAttribute<'c>>,
    alias_scopes: Option<ArrayAttribute<'c>>,
    noalias_scopes: Option<ArrayAttribute<'c>>,
    tbaa: Option<ArrayAttribute<'c>>,
}

impl<'c> LoadStoreOptions<'c> {
    /// Creates load/store options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an alignment.
    pub fn align(mut self, align: Option<IntegerAttribute<'c>>) -> Self {
        self.align = align;
        self
    }

    /// Sets a volatile flag.
    pub fn volatile(mut self, volatile: bool) -> Self {
        self.volatile = volatile;
        self
    }

    /// Sets a nontemporal flag.
    pub fn nontemporal(mut self, nontemporal: bool) -> Self {
        self.nontemporal = nontemporal;
        self
    }

    /// Sets access groups.
    pub fn access_groups(mut self, access_groups: Option<ArrayAttribute<'c>>) -> Self {
        self.access_groups = access_groups;
        self
    }

    /// Sets alias scopes.
    pub fn alias_scopes(mut self, alias_scopes: Option<ArrayAttribute<'c>>) -> Self {
        self.alias_scopes = alias_scopes;
        self
    }

    /// Sets noalias scopes.
    pub fn nonalias_scopes(mut self, noalias_scopes: Option<ArrayAttribute<'c>>) -> Self {
        self.noalias_scopes = noalias_scopes;
        self
    }

    /// Sets TBAA metadata.
    pub const fn tbaa(mut self, tbaa: ArrayAttribute<'c>) -> Self {
        self.tbaa = Some(tbaa);
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

        if self.volatile {
            attributes.push((
                Identifier::new(context, "volatile_"),
                Attribute::unit(context),
            ));
        }

        if self.nontemporal {
            attributes.push((
                Identifier::new(context, "nontemporal"),
                Attribute::unit(context),
            ));
        }

        if let Some(access_groups) = self.access_groups {
            attributes.push((
                Identifier::new(context, "access_groups"),
                access_groups.into(),
            ));
        }

        if let Some(alias_scopes) = self.alias_scopes {
            attributes.push((
                Identifier::new(context, "alias_scopes"),
                alias_scopes.into(),
            ));
        }

        if let Some(noalias_scopes) = self.noalias_scopes {
            attributes.push((
                Identifier::new(context, "noalias_scopes"),
                noalias_scopes.into(),
            ));
        }

        if let Some(tbaa) = self.tbaa {
            attributes.push((Identifier::new(context, "tbaa"), tbaa.into()));
        }

        attributes
    }
}
