//! # Standard symmetric eigenvalue problem: Eigenvalues of 1D Laplacian
//!
//! See examples/ex01_rusty for a more Rusty implementation
//!
//! Example from
//! https://slepc.upv.es/documentation/current/src/eps/tutorials/ex1.c.html
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int};

fn check_err(ierr: i32, function_name: &str) {
    if ierr != 0 {
        println!("error code {} from {}", ierr, function_name);
    }
}

// Requested tolerance
const EPS_TOL: slepc_sys::PetscReal = 1e-5;
// Maximum number of iterations
const EPS_MAXIT: slepc_sys::PetscInt = 10000;
// Requested eigenvalues
const EPS_NEV: slepc_sys::PetscInt = 2;

fn main() {
    // Set openblas num threads to 1, otherwise it might
    // conflict with mpi parallelization
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    // Command line arguments
    let argv = std::env::args().collect::<Vec<String>>();
    let mut c_argv = argv
        .into_iter()
        .map(|arg| CString::new(arg).expect("CString::new failed").into_raw())
        .collect::<Vec<*mut c_char>>();
    // Without this, a segmentation fault occurs ...
    c_argv.push(std::ptr::null_mut());
    let c_argv_ptr = &mut c_argv.as_mut_ptr();
    let c_argc_ptr = &mut (c_argv.len() as c_int);

    // Parameters
    let n = 10000;
    let local_rows: Option<slepc_sys::PetscInt> = None;
    let local_cols: Option<slepc_sys::PetscInt> = None;
    let global_rows: Option<slepc_sys::PetscInt> = Some(n);
    let global_cols: Option<slepc_sys::PetscInt> = Some(n);

    // Uninitialized pointers
    let mut mat_p_ = MaybeUninit::uninit();
    let mut i_start = MaybeUninit::<slepc_sys::PetscInt>::uninit();
    let mut i_end = MaybeUninit::<slepc_sys::PetscInt>::uninit();
    let mut xr = MaybeUninit::uninit();
    let mut xi = MaybeUninit::uninit();
    let mut eps_ = MaybeUninit::uninit();

    let mut its_ = MaybeUninit::uninit();
    let mut type_ = MaybeUninit::uninit();
    let mut nev_ = MaybeUninit::uninit();
    let mut tol_ = MaybeUninit::uninit();
    let mut maxit_ = MaybeUninit::uninit();
    let mut nconv_ = MaybeUninit::uninit();
    let mut kr_ = MaybeUninit::<slepc_sys::PetscScalar>::uninit();
    let mut ki_ = MaybeUninit::<slepc_sys::PetscScalar>::uninit();
    let mut error_ = MaybeUninit::<slepc_sys::PetscScalar>::uninit();

    unsafe {
        // Initialize slepc
        let ierr =
            slepc_sys::SlepcInitialize(c_argc_ptr, c_argv_ptr, std::ptr::null(), std::ptr::null());
        check_err(ierr, "SlepcInitialize");

        // Print hello world
        let msg = CString::new("Hello from SLEPc\n").unwrap();
        let ierr = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        check_err(ierr, "PetscPrintf");

        // Initialize Matrix
        let ierr = slepc_sys::MatCreate(slepc_sys::PETSC_COMM_WORLD, mat_p_.as_mut_ptr());
        check_err(ierr, "MatCreate");
        let mut mat_p: slepc_sys::Mat = mat_p_.assume_init();

        // ----------------------------------------------
        //               Build Matrix
        // ----------------------------------------------

        // Setup Matrix
        let ierr = slepc_sys::MatSetSizes(
            mat_p,
            local_rows.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            local_cols.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            global_rows.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            global_cols.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
        );
        check_err(ierr, "MatSetSizes");
        let ierr = slepc_sys::MatSetFromOptions(mat_p);
        check_err(ierr, "MatSetFromOptions");
        let ierr = slepc_sys::MatSetUp(mat_p);
        check_err(ierr, "MatSetUp");

        // Assemble matrix
        let ierr = slepc_sys::MatGetOwnershipRange(mat_p, i_start.as_mut_ptr(), i_end.as_mut_ptr());
        check_err(ierr, "MatGetOwnershipRange");

        // By default the values, v, are row-oriented.
        for i in i_start.assume_init()..i_end.assume_init() {
            let idxm: &[slepc_sys::PetscInt] = &[i];
            let (idxn, v): (Vec<slepc_sys::PetscInt>, &[slepc_sys::PetscScalar]) = if i == 0 {
                (vec![i, i + 1], &[2., -1.])
            } else if i == n - 1 {
                (vec![i - 1, i], &[-1., 2.])
            } else {
                (vec![i - 1, i, i + 1], &[-1., 2., -1.])
            };
            let m = idxm.len();
            let n = idxn.len();
            assert_eq!(v.len(), m * n);
            let ierr = slepc_sys::MatSetValues(
                mat_p,
                m as slepc_sys::PetscInt,
                idxm.as_ptr(),
                n as slepc_sys::PetscInt,
                idxn.as_ptr(),
                v.as_ptr() as *mut _,
                slepc_sys::InsertMode::INSERT_VALUES,
            );
            check_err(ierr, "MatSetValues");
        }

        let ierr =
            slepc_sys::MatAssemblyBegin(mat_p, slepc_sys::MatAssemblyType::MAT_FINAL_ASSEMBLY);
        check_err(ierr, "MatAssemblyBegin");
        let ierr = slepc_sys::MatAssemblyEnd(mat_p, slepc_sys::MatAssemblyType::MAT_FINAL_ASSEMBLY);
        check_err(ierr, "MatAssemblyEnd");

        // Create vectors
        let ierr = slepc_sys::MatCreateVecs(mat_p, std::ptr::null_mut(), xr.as_mut_ptr());
        check_err(ierr, "MatCreateVecs");
        let ierr = slepc_sys::MatCreateVecs(mat_p, std::ptr::null_mut(), xi.as_mut_ptr());
        check_err(ierr, "MatCreateVecs");

        // ----------------------------------------------
        //              Create the eigensolver
        // ----------------------------------------------

        // Create eigensolve context
        let ierr = slepc_sys::EPSCreate(slepc_sys::PETSC_COMM_WORLD, eps_.as_mut_ptr());
        check_err(ierr, "EPSCreate");
        let mut eps = eps_.assume_init();

        // Set operators
        let ierr = slepc_sys::EPSSetOperators(eps, mat_p, std::ptr::null_mut());
        check_err(ierr, "EPSSetOperators");

        // Set specific solver options
        let ierr = slepc_sys::EPSSetTolerances(eps, EPS_TOL, EPS_MAXIT);
        check_err(ierr, "EPSSetTolerances");

        let ierr = slepc_sys::EPSSetDimensions(
            eps,
            EPS_NEV,
            slepc_sys::PETSC_DECIDE_INTEGER,
            slepc_sys::PETSC_DECIDE_INTEGER,
        );
        check_err(ierr, "EPSSetDimensions");

        // Set solver parameters at runtime
        let ierr = slepc_sys::EPSSetFromOptions(eps);
        check_err(ierr, "EPSSetFromOptions");

        // ----------------------------------------------
        //             Solve the eigensystem
        // ----------------------------------------------
        let ierr = slepc_sys::EPSSolve(eps);
        check_err(ierr, "EPSSolve");

        // Display some information
        let _ = slepc_sys::EPSGetIterationNumber(eps, its_.as_mut_ptr());
        let _ = slepc_sys::EPSGetType(eps, type_.as_mut_ptr());
        let _ = slepc_sys::EPSGetDimensions(
            eps,
            nev_.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let _ = slepc_sys::EPSGetTolerances(eps, tol_.as_mut_ptr(), maxit_.as_mut_ptr());
        let msg = CString::new(format!(
            "Number of iterations of the method: {} \n",
            its_.assume_init()
        ))
        .unwrap();
        let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        let c_str: &std::ffi::CStr = std::ffi::CStr::from_ptr(type_.assume_init());
        let msg = CString::new(format!("Solution method: {} \n", c_str.to_str().unwrap())).unwrap();
        let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        let msg = CString::new(format!(
            "Number of requested eigenvalues: {} \n",
            nev_.assume_init()
        ))
        .unwrap();
        let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        let msg = CString::new(format!(
            "Stopping condition: tol={:e}, maxit={} \n",
            tol_.assume_init(),
            maxit_.assume_init()
        ))
        .unwrap();
        let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());

        // Get number of converged eigenpairs
        let _ = slepc_sys::EPSGetConverged(eps, nconv_.as_mut_ptr());
        let msg = CString::new(format!(
            "Number of converged eigenvalues: {} \n",
            nconv_.assume_init()
        ))
        .unwrap();
        let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());

        if nconv_.assume_init() > 0 {
            // Display eigenvalues and relative errors
            let msg = CString::new(
                "\n           k          ||Ax-kx||/||kx||\n  ----------------- ------------------\n",
            )
            .unwrap();
            let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());

            for i in 0..nconv_.assume_init() {
                // Get converged eigenpairs
                let ierr = slepc_sys::EPSGetEigenpair(
                    eps,
                    i,
                    kr_.as_mut_ptr(),
                    ki_.as_mut_ptr(),
                    xr.assume_init(),
                    xi.assume_init(),
                );
                check_err(ierr, "EPSGetEigenpair");

                // Compute the relative error associated to each eigenpair
                let ierr = slepc_sys::EPSComputeError(
                    eps,
                    i,
                    slepc_sys::EPSErrorType::EPS_ERROR_RELATIVE,
                    error_.as_mut_ptr(),
                );
                check_err(ierr, "EPSComputeError");

                let re = kr_.assume_init();
                let im = ki_.assume_init();

                let msg = CString::new(format!(
                    "{:9.6} {:9.6}i {:12.1e} \n",
                    re,
                    im,
                    error_.assume_init()
                ))
                .unwrap();
                let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
            }
            let msg = CString::new("\n").unwrap();
            let _ = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        }

        // ----------------------------------------------
        //             Finalize
        // ----------------------------------------------

        // TODO: This errors with Invalid Pointer
        // let _ = slepc_sys::MatDestroy(mat_p as *mut _);
        // let _ = slepc_sys::EPSDestroy(eps as *mut _);

        let ierr = slepc_sys::VecDestroy(&mut xr.assume_init() as *mut _);
        check_err(ierr, "VecDestroy");

        let ierr = slepc_sys::VecDestroy(&mut xi.assume_init() as *mut _);
        check_err(ierr, "VecDestroy");

        let ierr = slepc_sys::MatDestroy(&mut mat_p as *mut _);
        check_err(ierr, "MatDestroy");

        let ierr = slepc_sys::EPSDestroy(&mut eps as *mut _);
        check_err(ierr, "EPSDestroy");

        // Finalize slepc
        let ierr = slepc_sys::SlepcFinalize();
        check_err(ierr, "SlepcFinalize");
    };
}
