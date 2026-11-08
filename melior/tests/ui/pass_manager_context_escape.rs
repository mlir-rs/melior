use melior::{
    Context,
    ir::{Location, Module},
    pass::PassManager,
};

fn main() {
    let pass_manager = {
        let context = Context::new();

        PassManager::new(&context)
    };

    let context = Context::new();
    let _ = pass_manager.run(&mut Module::new(Location::unknown(&context)));
}
