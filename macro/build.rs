use std::{env, error::Error, path::Path, process::Command, str};

const LLVM_MAJOR_VERSION: usize = 20;

fn main() -> Result<(), Box<dyn Error>> {
    let version_variable = format!("MLIR_SYS_{LLVM_MAJOR_VERSION}0_PREFIX");

    println!("cargo:rerun-if-env-changed={version_variable}");
    println!(
        "cargo:rustc-env=LLVM_INCLUDE_DIRECTORY={}",
        // spell-checker: disable-next-line
        llvm_config("--includedir", &version_variable)?
    );

    Ok(())
}

fn llvm_config(
    argument: &str,
    version_variable: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let prefix = env::var(version_variable)
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let call = format!(
        "{} --link-static {}",
        prefix.join("llvm-config").display(),
        argument
    );

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}
