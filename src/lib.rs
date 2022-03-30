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
//! ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90
//! make all check
//! ```
//! (You can choose a custom arch name, or build without fc, i.e. `--with-fc=0`,
//! or build together with lapack, i.e. --download-f2cblaslapack)
//!
//! Then export directory, arch name and library path, i.e.
//! ```text
//! export PETSC_DIR=${HOME}/local/petsc-3.16.4
//! export PETSC_ARCH=linux-gnu-real-debug
//! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PETSC_DIR}/${PETSC_ARCH}/lib
//! ```
//!
//! ### `PETSc` Complex
//! ```text
//! export PETSC_ARCH=linux-gnu-complex-debug
//! ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-scalar-type=complex
//! make all check
//! ```
//! (You can choose a custom arch name, or build without fc, i.e. `--with-fc=0`,
//! or build together with lapack, i.e. --download-f2cblaslapack)
//!
//! - *Be careful that LD_LIBRARY_PATH has no old links to a build without complex type!*
//!
//! - Turn on feature `scalar_complex`!
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
#![warn(clippy::pedantic)]
pub mod eps;
#[cfg(feature = "gnuplot")]
pub mod gp;
pub mod ksp;
pub mod mat;
pub mod mat_shell;
pub mod pc;
pub mod st;
pub mod vec;
pub mod world;

#[derive(Debug, Clone)]
pub struct PetscError {
    pub(crate) ierr: slepc_sys::PetscErrorCode,
}

impl std::fmt::Display for PetscError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error code {}.", self.ierr)
    }
}

/// `PETSc` result
pub type Result<T> = std::result::Result<T, PetscError>;

/// Check  `PETSc` Error
/// Todo: More checks
///
/// # Errors
/// If ierr is non zero
pub fn check_error(ierr: slepc_sys::PetscErrorCode) -> Result<()> {
    if ierr == 0 {
        Ok(())
    } else {
        Err(PetscError { ierr })
    }
}

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
