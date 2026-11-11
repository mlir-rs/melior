//! The Toy compiler command.

use clap::{Parser, ValueEnum};
use melior::ir::Module;
use melior_kaleidoscope::{Action, create_context, generate, parse::parse, process, run_jit};
use std::{error::Error, fs::read_to_string, path::PathBuf, process::exit};

#[derive(Parser)]
#[command(version, about = "The Toy compiler")]
struct Arguments {
    /// An input file in the Toy language or MLIR for the `.mlir` extension.
    input: PathBuf,
    /// A kind of output.
    #[arg(long, value_enum)]
    emit: Emit,
    /// Enables optimizations.
    #[arg(long = "opt")]
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
    /// JIT compilation and execution of the main function.
    Jit,
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
    let file_name = arguments.input.display().to_string();

    let action = match arguments.emit {
        Emit::Ast => {
            print!("{}", parse(&source, &file_name)?);
            return Ok(());
        }
        Emit::Mlir => Action::Mlir,
        Emit::MlirAffine => Action::MlirAffine,
        Emit::MlirLlvm => Action::MlirLlvm,
        Emit::Jit => Action::Jit,
    };

    let context = create_context();
    let mut module = if arguments
        .input
        .extension()
        .is_some_and(|extension| extension == "mlir")
    {
        Module::parse(&context, &source).ok_or(melior_kaleidoscope::Error::ModuleParse)?
    } else {
        generate(&context, &parse(&source, &file_name)?)?
    };

    process(&context, &mut module, action, arguments.optimize)?;

    if action == Action::Jit {
        run_jit(&module, arguments.optimize)?;
    } else {
        print!("{}", module.as_operation());
    }

    Ok(())
}
