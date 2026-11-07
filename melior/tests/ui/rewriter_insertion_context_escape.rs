use melior::{
    Context, IrRewriter,
    ir::{Location, Module, operation::OperationBuilder},
};

fn main() {
    let context = Context::new();
    let module = Module::new(Location::unknown(&context));

    {
        let other_context = Context::new();
        let rewriter = IrRewriter::new(&other_context);
        let rewriter = rewriter.as_rewriter_base();

        rewriter.set_insertion_point_to_end(module.body());
        rewriter.insert(
            OperationBuilder::new("foo.bar", Location::unknown(&other_context))
                .build()
                .unwrap(),
        );
    }

    let _ = module.as_operation().to_string();
}
