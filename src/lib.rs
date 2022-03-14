//! # `slepc-rs`: `SLEPc` rust bindings
//!
//! Currently, this repository is only to demonstrate
//! how `SLEPc` can be called from Rust.
//!
//! ## Dependencies
//! - `clang`
//! - `mpi`
//! - `petsc` and `slepc`, see Installation section
//!
//! ## Installation
//!
//! ### `PETSc`
//! Download [download `PETSc`](https://petsc.org/release/download/). This
//! crate is tested with v.`3.16.4`.
//!
//! Install petsc via
//! ```text
//! export PETSC_ARCH=linux-gnu-real-debug
//! ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-f2cblaslapack
//! make all check
//! ```
//! (You can choose a custom arch name, or build without fc, i.e. `--with-fc=0`)
//!
//! Then export directory, arch name and library path, i.e.
//! ```text
//! export PETSC_DIR=${HOME}/local/petsc-3.16.4
//! export PETSC_ARCH=linux-gnu-real-debug
//! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PETSC_DIR}/${PETSC_ARCH}/lib
//! ```
//!
//! ### `SLEPc`
//! Download [download `SLEPc`](https://slepc.upv.es/download/). This
//! crate is tested with v.`3.16.2`.
//!
//! Install slepc via
//! ```text
//! ./configure
//! make all check
//! ```
//! Then export directory and library path, i.e.
//! ```text
//! export SLEPC_DIR=${HOME}/local/slepc-3.16.2
//! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SLEPC_DIR}/${PETSC_ARCH}/lib
//! ```
//!
//! ## Other similar crates
//! - <https://gitlab.com/petsc/petsc-rs>
//! - <https://github.com/tflovorn/slepc-sys>
//!
//! ## TODO
//! - Error Handling
//!
//! ## Documentation
//! - <https://slepc.upv.es/documentation/slepc.pdf>
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
pub mod eigensolver;
#[cfg(feature = "with_gnuplot")]
pub mod gnuplot;
pub mod linear_system;
pub mod matrix;
pub mod matrix_shell;
pub mod preconditioner;
pub mod spectral_transform;
pub mod vector;
pub mod world;
// Reimport all `slepc_sys` routines
pub use slepc_sys;

// From `rsmpi` crate
pub(crate) unsafe fn with_uninitialized<F, U, R>(f: F) -> (R, U)
where
    F: FnOnce(*mut U) -> R,
{
    let mut uninitialized = std::mem::MaybeUninit::uninit();
    let res = f(uninitialized.as_mut_ptr());
    (res, uninitialized.assume_init())
}

// From `rsmpi` crate
pub(crate) unsafe fn with_uninitialized2<F, U, V, R>(f: F) -> (R, U, V)
where
    F: FnOnce(*mut U, *mut V) -> R,
{
    let mut uninitialized0 = std::mem::MaybeUninit::uninit();
    let mut uninitialized1 = std::mem::MaybeUninit::uninit();
    let res = f(uninitialized0.as_mut_ptr(), uninitialized1.as_mut_ptr());
    (
        res,
        uninitialized0.assume_init(),
        uninitialized1.assume_init(),
    )
}
