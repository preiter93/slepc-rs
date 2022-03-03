pub mod petsc_sys;
pub mod slepc_sys;

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::vec::Vec as VecR;

fn main() {
    use crate::petsc_sys::{PetscPrintf, PETSC_COMM_WORLD};
    use crate::slepc_sys::{SlepcFinalize, SlepcInitialize};

    println!("Hello, world!");
    let argv = std::env::args().collect::<VecR<String>>();
    let argc = argv.len();

    let mut c_argc = argc as c_int;
    let mut c_argv = argv
        .into_iter()
        .map(|arg| CString::new(arg).unwrap().into_raw())
        .collect::<VecR<*mut c_char>>();
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

        let msg = CString::new("Hello from SLEPc and PETSc\n").unwrap();

        let ierr = PetscPrintf(PETSC_COMM_WORLD, msg.as_ptr());
        if ierr != 0 {
            println!("error code {} from PetscPrintf", ierr);
        }

        let ierr = SlepcFinalize();
        if ierr != 0 {
            println!("error code {} from SlepcFinalize", ierr);
        }
    };
}

// fn main() {
//     println!("Hello, world!");

//     let argv = std::env::args().collect::<VecR<String>>();
//     let argc = argv.len();

//     let mut c_argc = argc as c_int;
//     let mut c_argv = argv
//         .into_iter()
//         .map(|arg| CString::new(arg).unwrap().into_raw())
//         .collect::<VecR<*mut c_char>>();
//     let mut c_argv_ptr = c_argv.as_mut_ptr();

//     unsafe {
//         let ierr = PetscInitialize(
//             &mut c_argc,
//             &mut c_argv_ptr,
//             std::ptr::null(),
//             std::ptr::null(),
//         );

//         if ierr != 0 {
//             println!("error code {} from PetscInitialize", ierr);
//         }

//         let ierr = PetscFinalize();
//         if ierr != 0 {
//             println!("error code {} from PetscFinalize", ierr);
//         }
//     };
// }
