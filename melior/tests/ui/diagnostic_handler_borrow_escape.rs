use melior::{Context, ir::Module};

fn main() {
    let context = Context::new();

    {
        let mut messages = vec![];

        context.attach_diagnostic_handler(|diagnostic| {
            messages.push(diagnostic.to_string());
            true
        });
    }

    Module::parse(&context, "foo");
}
