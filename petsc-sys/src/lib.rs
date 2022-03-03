#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_copy_implementations)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(deref_nullptr)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const PETSC_DECIDE_INTEGER: PetscInt = PETSC_DECIDE as PetscInt;
pub const PETSC_DEFAULT_INTEGER: PetscInt = PETSC_DEFAULT as PetscInt;
pub const PETSC_DEFAULT_REAL: PetscReal = PETSC_DEFAULT as PetscReal;
