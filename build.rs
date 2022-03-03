extern crate bindgen;
extern crate build_probe_mpi;

use std::env;
use std::path::PathBuf;
/*
fn build_slepc() {
    // Petsc
    let petsc_include_dir: PathBuf = [env::var("PETSC_DIR").unwrap(), String::from("include")]
        .iter()
        .collect();

    let petsc_arch_include_dir: PathBuf = [
        env::var("PETSC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("include"),
    ]
    .iter()
    .collect();

    let petsc_lib_dir: PathBuf = [
        env::var("PETSC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    // Allow user to set SLEPc paths from environment variables.
    let slepc_include_dir: PathBuf = [env::var("SLEPC_DIR").unwrap(), String::from("include")]
        .iter()
        .collect();

    let slepc_arch_include_dir: PathBuf = [
        env::var("SLEPC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("include"),
    ]
    .iter()
    .collect();

    let slepc_lib_dir: PathBuf = [
        env::var("SLEPC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    println!("cargo:rustc-link-search={}", slepc_lib_dir.display());
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}",
        slepc_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=slepc");
}*/

fn main() {
    // Allow user to set PETSc paths from environment variables.
    let petsc_include_dir: PathBuf = [env::var("PETSC_DIR").unwrap(), String::from("include")]
        .iter()
        .collect();

    let petsc_arch_include_dir: PathBuf = [
        env::var("PETSC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("include"),
    ]
    .iter()
    .collect();

    let petsc_lib_dir: PathBuf = [
        env::var("PETSC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    println!("cargo:rustc-link-search={}", petsc_lib_dir.display());
    // the binary will look for the petsc lib in the directory pointed at by LD_LIBRARY_PATH
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}",
        petsc_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=petsc");

    // Allow user to set SLEPc paths from environment variables.
    let slepc_include_dir: PathBuf = [env::var("SLEPC_DIR").unwrap(), String::from("include")]
        .iter()
        .collect();

    let slepc_arch_include_dir: PathBuf = [
        env::var("SLEPC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("include"),
    ]
    .iter()
    .collect();

    let slepc_lib_dir: PathBuf = [
        env::var("SLEPC_DIR").unwrap(),
        env::var("PETSC_ARCH").unwrap(),
        String::from("lib"),
    ]
    .iter()
    .collect();

    println!("cargo:rustc-link-search={}", slepc_lib_dir.display());
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}",
        slepc_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=slepc");
    // println!("cargo:rustc-link-lib=petsc");

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

    // Set up builder with MPI and PETSc library and include paths.
    let mut builder_petsc = bindgen::Builder::default();
    let mut builder_slepc = bindgen::Builder::default();

    for dir in &mpi_lib.lib_paths {
        builder_petsc = builder_petsc.clang_arg(format!("-L{}", dir.display()));
        builder_slepc = builder_slepc.clang_arg(format!("-L{}", dir.display()));
    }

    for dir in &mpi_lib.include_paths {
        builder_petsc = builder_petsc.clang_arg(format!("-I{}", dir.display()));
        builder_slepc = builder_slepc.clang_arg(format!("-I{}", dir.display()));
    }

    builder_petsc = builder_petsc.clang_arg(format!("-L{}", petsc_lib_dir.display()));
    builder_petsc = builder_petsc.clang_arg(format!("-I{}", petsc_include_dir.display()));
    builder_petsc = builder_petsc.clang_arg(format!("-I{}", petsc_arch_include_dir.display()));

    builder_slepc = builder_slepc.clang_arg(format!("-L{}", petsc_lib_dir.display()));
    builder_slepc = builder_slepc.clang_arg(format!("-I{}", petsc_include_dir.display()));
    builder_slepc = builder_slepc.clang_arg(format!("-I{}", petsc_arch_include_dir.display()));
    builder_slepc = builder_slepc.clang_arg(format!("-L{}", slepc_lib_dir.display()));
    builder_slepc = builder_slepc.clang_arg(format!("-I{}", slepc_include_dir.display()));
    builder_slepc = builder_slepc.clang_arg(format!("-I{}", slepc_arch_include_dir.display()));

    let bindings_petsc = builder_petsc
        // The input header we would like to generate
        // bindings for.
        .header("src/petsc_sys/petsc_wrapper.h")
        // PETSc defines FP_* things twice and we will get errors
        .blacklist_item("FP\\w*")
        // Tell cargo to not mangle the function names
        .trust_clang_mangling(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .parse_callbacks(Box::new(ignored_macros))
        .rustfmt_bindings(true)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write out PETSc bindings.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings_petsc
        .write_to_file(out_path.join("bindings_petsc.rs"))
        .expect("Couldn't write PETSc bindings");

    let bindings_slepc = builder_slepc
        // The input header we would like to generate
        // bindings for.
        .header("src/slepc_sys/slepc_wrapper.h")
        // PETSc defines FP_* things twice and we will get errors
        .blacklist_item("FP\\w*")
        // Tell cargo to not mangle the function names
        .trust_clang_mangling(false)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // .parse_callbacks(Box::new(ignored_macros))
        .rustfmt_bindings(true)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write out PETSc bindings.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings_slepc
        .write_to_file(out_path.join("bindings_slepc.rs"))
        .expect("Couldn't write SLEPc bindings");
}
