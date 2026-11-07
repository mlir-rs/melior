use melior::{Context, ir::Location};

fn main() {
    let location = {
        let context = Context::new();

        Location::call_site(Location::unknown(&context), Location::unknown(&context))
    };

    let _ = location.to_string();
}
