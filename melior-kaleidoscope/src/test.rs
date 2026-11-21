use crate::{
    compile::{Action, compile},
    error::Error,
    mlir_gen::generate,
    parse::parse,
};
use melior::{Context, ir::Module};

pub fn compile_source<'c>(
    context: &'c Context,
    source: &str,
    action: Action,
    optimize: bool,
) -> Result<Module<'c>, Error> {
    let mut module = generate(context, &parse(source, "test.toy")?)?;

    compile(context, &mut module, action, optimize)?;

    Ok(module)
}
