#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VariadicKind {
    Simple {
        unfixed_seen: bool,
    },
    SameSize {
        unfixed_count: usize,
        preceding_simple_count: usize,
        preceding_variadic_count: usize,
    },
    AttributeSized,
}

impl VariadicKind {
    pub fn new(unfixed_count: usize, same_size: bool, attribute_sized: bool) -> Self {
        if unfixed_count <= 1 {
            Self::Simple {
                unfixed_seen: false,
            }
        } else if same_size {
            Self::SameSize {
                unfixed_count,
                preceding_simple_count: 0,
                preceding_variadic_count: 0,
            }
        } else if attribute_sized {
            Self::AttributeSized
        } else {
            unimplemented!()
        }
    }
}
