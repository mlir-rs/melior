use melior::{
    Context,
    ir::{OperationRef, r#type::TypeId},
    pass::{ExternalPass, PassManager, create_external},
};
use std::{cell::Cell, rc::Rc};

#[repr(align(8))]
struct PassId;

static PASS_ID: PassId = PassId;

fn main() {
    let context = Context::new();
    let manager = PassManager::new(&context);
    let counter = Rc::new(Cell::new(0));

    manager.add_pass(create_external(
        move |_: OperationRef, _: ExternalPass| counter.set(counter.get() + 1),
        TypeId::create(&PASS_ID),
        "name",
        "argument",
        "description",
        "",
        &[],
    ));
}
