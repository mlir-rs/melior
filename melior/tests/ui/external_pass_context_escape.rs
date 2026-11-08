use melior::{
    Context,
    ir::{Location, OperationRef, operation::OperationLike, r#type::TypeId},
    pass::{ExternalPass, PassManager, create_external},
};
use std::cell::RefCell;

#[repr(align(8))]
struct PassId;

static PASS_ID: PassId = PassId;

thread_local! {
    static LOCATION: RefCell<Option<Location<'static>>> = const { RefCell::new(None) };
}

fn main() {
    let context = Context::new();
    let manager = PassManager::new(&context);

    manager.add_pass(create_external(
        |operation: OperationRef<'static, '_>, _: ExternalPass| {
            LOCATION.with(|location| *location.borrow_mut() = Some(operation.location()));
        },
        TypeId::create(&PASS_ID),
        "name",
        "argument",
        "description",
        "",
        &[],
    ));
}
