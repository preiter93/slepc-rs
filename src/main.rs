//! Example from
//! https://slepc.upv.es/documentation/current/src/eps/tutorials/ex1.c.html
#![allow(dead_code)]
use std::ffi::CString;
// use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int};

fn ex0() {
    // Command line arguments
    let argv = std::env::args().collect::<Vec<String>>();
    let argc = argv.len();
    let mut c_argc = argc as c_int;
    let mut c_argv = argv
        .into_iter()
        .map(|arg| CString::new(arg).unwrap().into_raw())
        .collect::<Vec<*mut c_char>>();
    let mut c_argv_ptr = c_argv.as_mut_ptr();

    unsafe {
        // Initialize slepc
        let ierr = slepc_sys::SlepcInitialize(
            &mut c_argc,
            &mut c_argv_ptr,
            std::ptr::null(),
            std::ptr::null(),
        );
        if ierr != 0 {
            println!("error code {} from SlepcInitialize", ierr);
        }

        // Print hello world
        let msg = CString::new("Hello from SLEPc\n").unwrap();
        let ierr = slepc_sys::PetscPrintf(slepc_sys::PETSC_COMM_WORLD, msg.as_ptr());
        if ierr != 0 {
            println!("error code {} from PetscPrintf", ierr);
        }

        // Finalize slepc
        let ierr = slepc_sys::SlepcFinalize();
        if ierr != 0 {
            println!("error code {} from SlepcFinalize", ierr);
        }
    };
}

fn check_err(ierr: i32, function_name: &str) {
    if ierr != 0 {
        println!("error code {} from {}", ierr, function_name);
    }
}

use slepc_rs::eigensolver::SlepcEps;
use slepc_rs::matrix::PetscMat;
use slepc_rs::world::SlepcWorld;

// Requested tolerance
const EPS_TOL: slepc_sys::PetscReal = 1e-5;
// Maximum number of iterations
const EPS_MAXIT: slepc_sys::PetscInt = 10000;
// Requested eigenvalues
const EPS_NEV: slepc_sys::PetscInt = 2;

fn main() {
    // Parameters
    let n = 1000;

    println!("Hello World");
    let world = SlepcWorld::initialize();

    // ----------------------------------------------
    //               Build Matrix
    // ----------------------------------------------

    let mut mat = PetscMat::create(&world);

    // (local_row, local_col, global_row, global_col)
    mat.set_sizes(None, None, Some(n), Some(n));
    mat.set_from_options();
    mat.set_up();

    // Rows are split to different processors
    let (i_start, i_end) = mat.get_ownership_range();

    // Construct laplacian
    let addv = slepc_sys::InsertMode::INSERT_VALUES;
    for i in i_start..i_end {
        let idxm: &[slepc_sys::PetscInt] = &[i];
        let (idxn, v): (Vec<slepc_sys::PetscInt>, &[slepc_sys::PetscScalar]) = if i == 0 {
            (vec![i, i + 1], &[2., -1.])
        } else if i == n - 1 {
            (vec![i - 1, i], &[-1., 2.])
        } else {
            (vec![i - 1, i, i + 1], &[-1., 2., -1.])
        };
        mat.set_values(idxm, &idxn, v, addv);
    }

    mat.assembly_begin(slepc_sys::MatAssemblyType::MAT_FINAL_ASSEMBLY);
    mat.assembly_end(slepc_sys::MatAssemblyType::MAT_FINAL_ASSEMBLY);

    let xr = mat.create_vec_left();
    let xi = mat.create_vec_left();

    // ----------------------------------------------
    //              Create the eigensolver
    // ----------------------------------------------

    let mut eps = SlepcEps::create(&world);
    eps.set_operators(Some(mat.as_raw()), None);
    eps.set_tolerances(Some(EPS_TOL), Some(EPS_MAXIT));
    eps.set_dimensions(Some(EPS_NEV), None, None);
    eps.set_from_options();

    // ----------------------------------------------
    //             Solve the eigensystem
    // ----------------------------------------------
    eps.solve();

    // Get values and print
    let eps_its = eps.get_iteration_number();
    let eps_type = eps.get_type();
    let eps_nconv = eps.get_converged();
    let (eps_nev, _eps_ncv) = eps.get_dimensions();
    let (eps_tol, eps_maxit) = eps.get_tolerances();

    world.print(&format!(
        " Number of iterations of the method: {} \n",
        eps_its
    ));
    world.print(&format!(" Solution method: {} \n", eps_type));
    world.print(&format!(" Number of requested eigenvalues: {} \n", eps_nev));
    world.print(&format!(
        " Number of converged eigenvalues: {} \n",
        eps_nconv
    ));
    world.print(&format!(
        " Stopping condition: tol={:e}, maxit={} \n",
        eps_tol, eps_maxit,
    ));

    world.print(
        "\n           k          ||Ax-kx||/||kx||\n  ----------------- ------------------\n",
    );
    for i in 0..eps_nconv {
        let (kr, ki) = eps.get_eigenpair(i, xr, xi);
        let error = eps.compute_error(i, slepc_sys::EPSErrorType::EPS_ERROR_RELATIVE);
        let (re, im) = (kr, ki);
        world.print(&format!("{:9.6} {:9.6}i {:12.2e} \n", re, im, error));
    }

    SlepcWorld::finalize();
}
/*pub mod petsc_sys;
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
}*/

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
