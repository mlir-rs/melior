use melior::{Context, ir::Location};

fn main() {
    let context = Context::new();
    let location = {
        let other_context = Context::new();

        Location::name(&context, "foo", Location::unknown(&other_context))
    };

    let _ = location.to_string();
}
