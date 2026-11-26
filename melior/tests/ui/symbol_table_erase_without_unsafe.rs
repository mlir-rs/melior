use melior::{
    Context,
    ir::{BlockLike, Module, SymbolTable},
};

fn main() {
    let context = Context::new();
    let module = Module::parse(&context, "func.func private @foo()").unwrap();
    let mut table = SymbolTable::new(module.as_operation()).unwrap();

    table
        .erase(module.body().first_operation().unwrap())
        .unwrap();
}
