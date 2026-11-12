//! The Toy compiler command.

use clap::{Parser, ValueEnum};
use melior::{ir::Module, llvm, utility::translate_module_to_llvm_ir};
use melior_kaleidoscope::{Action, compile, create_context, generate, parse::parse, run_jit};
use std::{error::Error, fs::read_to_string, path::PathBuf, process::exit};

#[derive(Parser)]
#[command(version, about = "The Toy compiler")]
struct Arguments {
    /// An input file in the Toy language or MLIR for the `.mlir` extension.
    input: PathBuf,
    /// A kind of output. The main function is JIT-compiled and executed if
    /// omitted.
    #[arg(long, value_enum)]
    emit: Option<Emit>,
    /// Enables optimizations.
    #[arg(long)]
    optimize: bool,
}

#[derive(Clone, Copy, Eq, PartialEq, ValueEnum)]
enum Emit {
    /// An AST dump.
    Ast,
    /// An MLIR dump.
    Mlir,
    /// An MLIR dump after affine lowering.
    MlirAffine,
    /// An MLIR dump after LLVM lowering.
    MlirLlvm,
    /// An LLVM IR dump.
    Llvm,
}

fn main() {
    let arguments = Arguments::parse();

    if let Err(error) = run(&arguments) {
        eprintln!("{}: {error}", arguments.input.display());
        exit(1);
    }
}

fn run(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
    let source = read_to_string(&arguments.input)?;
    let path = arguments.input.display().to_string();

    let action = match arguments.emit {
        Some(Emit::Ast) => {
            println!("{:#?}", parse(&source, &path)?);
            return Ok(());
        }
        Some(Emit::Mlir) => Action::Mlir,
        Some(Emit::MlirAffine) => Action::MlirAffine,
        Some(Emit::MlirLlvm) => Action::MlirLlvm,
        Some(Emit::Llvm) => Action::Llvm,
        None => Action::Jit,
    };

    let context = create_context();
    let mut module = if arguments
        .input
        .extension()
        .is_some_and(|extension| extension == "mlir")
    {
        Module::parse(&context, &source).ok_or(melior_kaleidoscope::Error::ModuleParse)?
    } else {
        generate(&context, &parse(&source, &path)?)?
    };

    compile(&context, &mut module, action, arguments.optimize)?;

    match action {
        Action::Mlir | Action::MlirAffine | Action::MlirLlvm => {
            print!("{}", module.as_operation())
        }
        Action::Llvm => print!(
            "{}",
            translate_module_to_llvm_ir(&module, &llvm::Context::new())?
        ),
        Action::Jit => run_jit(&module, arguments.optimize)?,
    }

    Ok(())
}
