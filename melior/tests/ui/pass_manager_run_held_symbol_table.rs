use melior::{
    Context,
    ir::{Location, Module, SymbolTable},
    pass::PassManager,
};

fn main() {
    let context = Context::new();
    let mut module = Module::new(Location::unknown(&context));
    let table = SymbolTable::new(module.as_operation()).unwrap();

    PassManager::new(&context).run(&mut module).unwrap();

    let _ = table.lookup("foo");
}
