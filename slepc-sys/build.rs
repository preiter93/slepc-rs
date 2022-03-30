use std::env;
use std::path::PathBuf;

fn main() {
    let petsc_lib_dir: PathBuf = [
        env::var("PETSC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    println!("cargo:rustc-link-search={}", petsc_lib_dir.display());
    println!("cargo:rustc-link-lib=petsc");

    let slepc_lib_dir: PathBuf = [
        env::var("SLEPC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    println!("cargo:rustc-link-search={}", slepc_lib_dir.display());
    println!("cargo:rustc-link-lib=slepc");
    println!("cargo:rustc-link-lib=petsc");

    // Find the system MPI library and headers,
    // in the same way as rsmpi/build.rs.
    let mpi_lib = match build_probe_mpi::probe() {
        Ok(mpi_lib) => mpi_lib,
        Err(errs) => {
            println!("Could not find MPI library for various reasons:\n");
            for (i, err) in errs.iter().enumerate() {
                println!("Reason #{}:\n{}\n", i, err);
            }
            panic!();
        }
    };

    for dir in &mpi_lib.lib_paths {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in &mpi_lib.libs {
        println!("cargo:rustc-link-lib={}", lib);
    }
}
