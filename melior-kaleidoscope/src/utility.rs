use crate::error::Error;
use core::iter::successors;
use melior::ir::{
    BlockLike, BlockRef, Module, OperationRef, RegionLike, Value,
    attribute::StringAttribute,
    operation::{OperationLike, OperationResult, WalkOrder, WalkResult},
};

// Checks that Toy operations other than legal ones are lowered.
pub fn check_lowering<'c>(
    module: &Module<'c>,
    is_legal: impl Fn(OperationRef<'c, '_>) -> bool,
) -> Result<(), Error> {
    let mut error = None;

    module
        .as_operation()
        .walk(WalkOrder::PreOrder, |operation| {
            if operation_name(operation).starts_with("toy.") && !is_legal(operation) {
                error = Some(Error::Lowering(format!(
                    "'{}' at {}",
                    operation_name(operation),
                    operation.location()
                )));

                WalkResult::Interrupt
            } else {
                WalkResult::Advance
            }
        });

    error.map_or(Ok(()), Err)
}

pub fn block_operations<'c: 'a, 'a>(
    block: impl BlockLike<'c, 'a>,
) -> impl Iterator<Item = OperationRef<'c, 'a>> {
    successors(block.first_operation(), |operation| {
        operation.next_in_block()
    })
}

pub fn body_block<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> Option<BlockRef<'c, 'a>> {
    operation.region(0).ok()?.first_block()
}

// Terminators of unregistered dialects are not recognized by MLIR, so the
// last operation in a block is taken as a terminator.
pub fn terminator<'c: 'a, 'a>(block: impl BlockLike<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    block_operations(block).last()
}

pub fn defining_operation<'c, 'a>(value: Value<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    OperationResult::try_from(value)
        .ok()
        .map(|result| result.owner())
}

pub fn is_operation<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>, name: &str) -> bool {
    operation.name().as_string_ref().as_str() == Ok(name)
}

pub fn operation_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> String {
    operation
        .name()
        .as_string_ref()
        .as_str()
        .unwrap_or_default()
        .into()
}

pub fn symbol_name<'c: 'a, 'a>(operation: impl OperationLike<'c, 'a>) -> Option<&'c str> {
    operation
        .attribute("sym_name")
        .ok()
        .and_then(|attribute| StringAttribute::try_from(attribute).ok())
        .map(|attribute| attribute.value())
}
