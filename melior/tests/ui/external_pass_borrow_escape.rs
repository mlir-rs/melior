use melior::{
    Context,
    ir::{OperationRef, r#type::TypeId},
    pass::{ExternalPass, PassManager, create_external},
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(align(8))]
struct PassId;

static PASS_ID: PassId = PassId;

fn main() {
    let context = Context::new();
    let manager = PassManager::new(&context);
    let counter = AtomicUsize::new(0);

    manager.add_pass(create_external(
        |_: OperationRef, _: ExternalPass| {
            counter.fetch_add(1, Ordering::Relaxed);
        },
        TypeId::create(&PASS_ID),
        "name",
        "argument",
        "description",
        "",
        &[],
    ));
}
