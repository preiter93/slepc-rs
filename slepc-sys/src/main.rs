use petsc_sys::{PetscPrintf, PETSC_COMM_WORLD};
use slepc_sys::{SlepcFinalize, SlepcInitialize};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

fn main() {
    use petsc_sys::Mat as Petsc_Mat;
    use petsc_sys::MatCreate;
    println!("Hello, world!");
    let argv = std::env::args().collect::<Vec<String>>();
    let argc = argv.len();
    println!("{:?}", argv);

    let mut c_argc = argc as c_int;
    let mut c_argv = argv
        .into_iter()
        .map(|arg| CString::new(arg).unwrap().into_raw())
        .collect::<Vec<*mut c_char>>();
    let mut c_argv_ptr = c_argv.as_mut_ptr();

    //let A: Petsc_Mat = std::mem::MaybeUninit::uninit();

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

        let mut mat_p = std::mem::MaybeUninit::uninit();
        let ierr = MatCreate(PETSC_COMM_WORLD, mat_p.as_mut_ptr());

        if ierr != 0 {
            println!("error code {} from MatCreate", ierr);
        }

        let _a: Petsc_Mat = mat_p.assume_init();

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

fn main2() {
    println!("Hello, world!");
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
