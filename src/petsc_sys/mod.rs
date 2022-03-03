#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_copy_implementations)]
#![allow(dead_code)]
#![allow(improper_ctypes)]
#![allow(deref_nullptr)]
include!(concat!(env!("OUT_DIR"), "/bindings_petsc.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int};
    use std::vec::Vec as VecR;

    #[test]
    fn test_petsc_compile() {
        let argv = std::env::args().collect::<VecR<String>>();
        let argc = argv.len();

        let mut c_argc = argc as c_int;
        let mut c_argv = argv
            .into_iter()
            .map(|arg| CString::new(arg).unwrap().into_raw())
            .collect::<VecR<*mut c_char>>();
        let mut c_argv_ptr = c_argv.as_mut_ptr();

        unsafe {
            let ierr = PetscInitialize(
                &mut c_argc,
                &mut c_argv_ptr,
                std::ptr::null(),
                std::ptr::null(),
            );

            if ierr != 0 {
                println!("error code {} from PetscInitialize", ierr);
            }

            let ierr = PetscFinalize();
            if ierr != 0 {
                println!("error code {} from PetscFinalize", ierr);
            }
        };
    }
}
