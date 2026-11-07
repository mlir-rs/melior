use melior::{
    Context,
    ir::{OperationRef, r#type::TypeId},
    pass::{ExternalPass, PassManager, create_external},
};

#[repr(align(8))]
struct PassId;

static PASS_ID: PassId = PassId;

fn main() {
    let context = Context::new();
    let manager = PassManager::new(&context);
    let operation_name = String::from("func.func");

    manager.add_pass(create_external(
        |_: OperationRef, _: ExternalPass| {},
        TypeId::create(&PASS_ID),
        "name",
        "argument",
        "description",
        &operation_name,
        &[],
    ));
}
