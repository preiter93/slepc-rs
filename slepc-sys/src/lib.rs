#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_copy_implementations)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(deref_nullptr)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub const PETSC_DETERMINE_INTEGER: PetscInt = PETSC_DETERMINE as PetscInt;
pub const PETSC_DETERMINE_REAL: PetscReal = PETSC_DETERMINE as PetscReal;

pub const PETSC_DECIDE_INTEGER: PetscInt = PETSC_DECIDE as PetscInt;
pub const PETSC_DECIDE_REAL: PetscReal = PETSC_DECIDE as PetscReal;

pub const PETSC_DEFAULT_INTEGER: PetscInt = PETSC_DEFAULT as PetscInt;
pub const PETSC_DEFAULT_REAL: PetscReal = PETSC_DEFAULT as PetscReal;

#[cfg(test)]
mod tests {
    use super::{SlepcFinalize, SlepcInitialize};
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    #[test]
    fn test_compile() {
        let argv = std::env::args().collect::<Vec<String>>();
        let argc = argv.len();

        let mut c_argc = argc as c_int;
        let mut c_argv = argv
            .into_iter()
            .map(|arg| CString::new(arg).unwrap().into_raw())
            .collect::<Vec<*mut c_char>>();
        let mut c_argv_ptr = c_argv.as_mut_ptr();

        unsafe {
            let ierr = SlepcInitialize(
                &mut c_argc,
                &mut c_argv_ptr,
                std::ptr::null(),
                std::ptr::null(),
            );
            if ierr != 0 {
                println!("error code {} from SlepcInitialize", ierr);
            }

            let ierr = SlepcFinalize();
            if ierr != 0 {
                println!("error code {} from SlepcFinalize", ierr);
            }
        };
    }
}
