use melior::{
    Context,
    ir::{Location, Module},
};

fn main() {
    let _module = {
        let context = Context::new();

        Module::new(Location::unknown(&context))
    };
}
