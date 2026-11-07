use melior::{
    Context,
    ir::{Attribute, Location},
};

fn main() {
    let context = Context::new();
    let location = {
        let other_context = Context::new();

        Location::fused(
            &context,
            &[Location::unknown(&context)],
            Attribute::unit(&other_context),
        )
    };

    let _ = location.to_string();
}
