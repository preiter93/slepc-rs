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
use slepc_rs::vector::PetscVec;
use slepc_rs::world::SlepcWorld;

// Requested tolerance
const EPS_TOL: slepc_sys::PetscReal = 1e-5;
// Maximum number of iterations
const EPS_MAXIT: slepc_sys::PetscInt = 50000;
// Requested eigenvalues
const EPS_NEV: slepc_sys::PetscInt = 2;

// fn my_matmult(mat: PetscMat, x: PetscVec, y: PetscVec) -> slepc_sys::PetscErrorCode {
//     todo!()
// }

// pub unsafe extern "C" fn my_matmult_raw(
//     _mat: slepc_sys::Mat,
//     x: slepc_sys::Vec,
//     y: slepc_sys::Vec,
// ) -> slepc_sys::PetscErrorCode {
//     todo!()
// }

// fn trampoline_type_a(
//     func: unsafe extern "C" fn(PetscMat, PetscVec, PetscVec) -> slepc_sys::PetscErrorCode,
// ) -> unsafe extern "C" fn(slepc_sys::Mat, slepc_sys::Vec, slepc_sys::Vec) -> slepc_sys::PetscErrorCode
// {
//     todo!()
// }

fn my_mat_mult(_mat: &PetscMat, x: &PetscVec, y: &mut PetscVec) {
    let x_view = x.get_array_read();
    let y_view_mut = y.get_array();
    let (i_start, i_end) = x.get_ownership_range();
    let n = i_end - i_start;
    assert!(
        n as usize == x_view.len(),
        "got {}, expected {}",
        n,
        x_view.len()
    );

    for (i, y_i) in y_view_mut.iter_mut().enumerate() {
        if i == 0 {
            *y_i = -2. * x_view[i] + 1. * x_view[i + 1];
        } else if i == n as usize - 1 {
            *y_i = 1. * x_view[i - 1] - 2. * x_view[i];
        } else {
            *y_i = 1. * x_view[i - 1] - 2. * x_view[i] + 1. * x_view[i + 1];
        }
    }
}

fn my_get_diagonal(_mat: &PetscMat, d: &mut PetscVec) {
    let d_view_mut = d.get_array();
    for (_, d_i) in d_view_mut.iter_mut().enumerate() {
        *d_i = 1.;
    }
}

// pub unsafe extern "C" fn my_matmult_raw(
//     mat: slepc_sys::Mat,
//     x: slepc_sys::Vec,
//     y: slepc_sys::Vec,
// ) -> slepc_sys::PetscErrorCode {
//     let r_mat = PetscMat::from_raw(mat);
//     let r_x = PetscVec::from_raw(x);
//     let mut r_y = PetscVec::from_raw(y);
//     my_matmult(&r_mat, &r_x, &mut r_y);
//     0
// }

// /// PETSc demands an unsafe C function with the signature
// ///     fn(Mat, Vec, Vec) -> PetscErrorCode
// /// but our matmul has the signature
// ///     fn(PetscMat, PetscVec, PetscVec)
// ///
// /// This trampoline macro creates the unsafe C function
// /// from a normal function.
// ///
// /// # Use
// /// ``` ignore
// /// trampoline_type_a!(my_matmult, my_matmult_raw);
// /// ```
// /// where my_matmult must be defined and my_matmult_raw is created.
// /// Then use my_matmult_raw in
// /// [`slepc_rs::matrix_shell::shell_set_operation_type_a`]
// macro_rules! trampoline_type_a {
//     (
//         $r: ident, $c: ident
//     ) => {
//         pub unsafe extern "C" fn $c(
//             mat: slepc_sys::Mat,
//             x: slepc_sys::Vec,
//             y: slepc_sys::Vec,
//         ) -> slepc_sys::PetscErrorCode {
//             let r_mat = PetscMat::from_raw(mat);
//             let r_x = PetscVec::from_raw(x);
//             let mut r_y = PetscVec::from_raw(y);
//             $r(&r_mat, &r_x, &mut r_y);
//             0
//         }
//     };
// }
use slepc_rs::{trampoline_type_a, trampoline_type_b};

fn main() {
    // Set openblas num threads to 1, otherwise it might
    // conflict with mpi parallelization
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    // Parameters
    let n = 1000;

    println!("Hello World");
    let world = SlepcWorld::initialize();

    // ----------------------------------------------
    //                  Shell Matrix
    // ----------------------------------------------
    let mat = PetscMat::create_shell(&world, n, n, Some(n), Some(n));

    let xr = mat.create_vec_left();
    let xi = mat.create_vec_left();

    let mut x = PetscVec::create(&world);
    x.set_sizes(None, Some(n));
    x.set_up();
    x.assembly_begin();
    x.assembly_end();

    let y = x.duplicate();
    y.assembly_begin();
    y.assembly_end();

    // Mat mult
    trampoline_type_b!(my_mat_mult, my_mat_mult_raw);
    mat.shell_set_operation_type_b(slepc_sys::MatOperation::MATOP_MULT, my_mat_mult_raw);
    // Mat mult transpose
    mat.shell_set_operation_type_b(
        slepc_sys::MatOperation::MATOP_MULT_TRANSPOSE,
        my_mat_mult_raw,
    );
    // Get diagonal
    trampoline_type_a!(my_get_diagonal, my_get_diagonal_raw);
    mat.shell_set_operation_type_a(
        slepc_sys::MatOperation::MATOP_GET_DIAGONAL,
        my_get_diagonal_raw,
    );

    // ----------------------------------------------
    //              Create the eigensolver
    // ----------------------------------------------

    let mut eps = SlepcEps::create(&world);
    eps.set_operators(Some(mat.as_raw()), None);
    eps.set_tolerances(Some(EPS_TOL), Some(EPS_MAXIT));
    eps.set_dimensions(Some(EPS_NEV), None, None);
    // eps.set_which_eigenpairs(slepc_sys::EPSWhich::EPS_LARGEST_REAL);
    eps.set_type("krylovschur");
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
        let (kr, ki) = eps.get_eigenpair(i, xr.as_raw(), xi.as_raw());
        let error = eps.compute_error(i, slepc_sys::EPSErrorType::EPS_ERROR_RELATIVE);
        let (re, im) = (kr, ki);
        world.print(&format!("{:9.6} {:9.6}i {:12.2e} \n", re, im, error));
    }

    // let (istart, iend) = xr.get_ownership_range();
    // let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>());
    // plot_gnu(&vec_vals);

    mat.destroy();
    xi.destroy();
    xr.destroy();
    eps.destroy();
    SlepcWorld::finalize();
}

fn main2() {
    // Parameters
    let n = 100;

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
    eps.set_which_eigenpairs(slepc_sys::EPSWhich::EPS_SMALLEST_REAL);
    // eps.set_type("arnoldi");
    eps.set_type("krylovschur");
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
        let (kr, ki) = eps.get_eigenpair(i, xr.as_raw(), xi.as_raw());
        let error = eps.compute_error(i, slepc_sys::EPSErrorType::EPS_ERROR_RELATIVE);
        let (re, im) = (kr, ki);
        world.print(&format!("{:9.6} {:9.6}i {:12.2e} \n", re, im, error));
    }

    let (istart, iend) = xr.get_ownership_range();
    let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>());
    plot_gnu(&vec_vals);
    SlepcWorld::finalize();
}

/// Plot line
/// # Example
/// Plot Petsc Vector
/// ```ìgnore
/// let (istart, iend) = xr.get_ownership_range();
/// let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>());
/// plot_gnu(&vec_vals);
/// ```
#[cfg(feature = "plot")]
fn plot_gnu(y: &[slepc_sys::PetscReal]) {
    use gnuplot::{Caption, Color, Figure};
    let x = (0..y.len())
        .map(|x| x as f64)
        .collect::<Vec<slepc_sys::PetscReal>>();
    let mut fg = Figure::new();
    fg.axes2d().lines(&x, y, &[Caption(""), Color("black")]);
    fg.show().unwrap();
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
