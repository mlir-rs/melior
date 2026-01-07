//! IR objects and builders.

mod affine_map;
pub mod attribute;
pub mod block;
mod identifier;
mod location;
mod module;
pub mod operation;
mod region;
pub mod r#type;
mod value;

pub use self::{
    affine_map::AffineMap,
    attribute::{Attribute, AttributeLike},
    block::{Block, BlockLike, BlockRef},
    identifier::Identifier,
    location::Location,
    module::Module,
    operation::{Operation, OperationRef},
    region::{Region, RegionLike, RegionRef},
    r#type::{ShapedTypeLike, Type, TypeLike},
    value::{Value, ValueLike},
};
