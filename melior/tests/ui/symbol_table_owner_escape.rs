use melior::{
    Context,
    ir::{Location, Module, SymbolTable},
};

fn main() {
    let context = Context::new();

    let table = {
        let module = Module::new(Location::unknown(&context));

        SymbolTable::new(module.as_operation()).unwrap()
    };

    let _ = table.lookup("foo");
}
