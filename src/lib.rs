//! # `slepc-rs`: SLEPc rust bindings
//!
//! Currently, this repository is only to demonstrate
//! how SLEPc can be called from Rust.
//!
//! ## Dependencies
//! - `clang`
//! - `mpi`
//! - `petsc` and `slepc`, see Installation section
//!
//! ## Installation
//!
//! ### PETSc
//! Download [download PETSc](https://petsc.org/release/download/). This
//! crate is tested with v.`3.16.4`.
//!
//! Install petsc via (you can choose a custom arch name, or build with fc, --with-fc=0)
//! ```text
//! export PETSC_ARCH=linux-gnu-real-debug
//! ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --download-f2cblaslapack
//! make all check
//! ```
//!
//! Then export directory, arch name and library path, i.e.
//! ```text
//! export PETSC_DIR=${HOME}/local/petsc-3.16.4
//! export PETSC_ARCH=linux-gnu-real-debug
//! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PETSC_DIR}/${PETSC_ARCH}/lib
//! ```
//!
//! ### SLEPc
//! Download [download SLEPc](https://slepc.upv.es/download/). This
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
