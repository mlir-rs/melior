use melior::{Context, ir::Module};

fn main() {
    let _module = {
        let context = Context::new();

        Module::parse(&context, "module {}")
    };
}
