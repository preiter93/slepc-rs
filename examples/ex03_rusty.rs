//! # Matrix free eigenvalues of 1D Laplacian
//!
//! Similar as `examples/ex03.rs` but here we use `slepc-rs` interfaces
//!
//! Example from
//! https://slepc.upv.es/documentation/current/src/eps/tutorials/ex3.c.html
//!
//! Note: Not MPI tested!
use slepc_rs::eps::SlepcEps;
use slepc_rs::mat::PetscMat;
use slepc_rs::vec::PetscVec;
use slepc_rs::world::SlepcWorld;
use slepc_rs::Result;

// Requested tolerance
const EPS_TOL: slepc_sys::PetscReal = 1e-5;
// Maximum number of iterations
const EPS_MAXIT: slepc_sys::PetscInt = 50000;
// Requested eigenvalues
const EPS_NEV: slepc_sys::PetscInt = 2;

/// Matmul 1d laplacian
fn my_mat_mult(_mat: &PetscMat, x: &PetscVec, y: &mut PetscVec) {
    let x_view = x.get_array_read().unwrap();
    let y_view_mut = y.get_array().unwrap();
    let (i_start, i_end) = x.get_ownership_range().unwrap();
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

/// Diagonal 1d laplacian
fn my_get_diagonal(_mat: &PetscMat, d: &mut PetscVec) {
    let d_view_mut = d.get_array().unwrap();
    for (_, d_i) in d_view_mut.iter_mut().enumerate() {
        *d_i = -2.;
    }
}

fn main() -> Result<()> {
    // Set openblas num threads to 1, otherwise it might
    // conflict with mpi parallelization
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    // Parameters
    let n = 1000;

    println!("Hello World");
    let world = SlepcWorld::initialize()?;

    // ----------------------------------------------
    //                  Shell Matrix
    // ----------------------------------------------
    let mat = PetscMat::create_shell::<u8>(&world, None, None, Some(n), Some(n), None)?;

    let xr = mat.create_vec_left()?;
    let xi = mat.create_vec_left()?;

    let mut x = PetscVec::create(&world)?;
    x.set_sizes(None, Some(n))?;
    x.set_up()?;
    x.assembly_begin()?;
    x.assembly_end()?;

    let y = x.duplicate()?;
    y.assembly_begin()?;
    y.assembly_end()?;

    // Mat mult
    mat.shell_set_operation_type_b(slepc_sys::MatOperation::MATOP_MULT, my_mat_mult)?;
    // Mat mult transpose
    mat.shell_set_operation_type_b(slepc_sys::MatOperation::MATOP_MULT_TRANSPOSE, my_mat_mult)?;
    // Get diagonal
    mat.shell_set_operation_type_a(slepc_sys::MatOperation::MATOP_GET_DIAGONAL, my_get_diagonal)?;

    // ----------------------------------------------
    //              Create the eigensolver
    // ----------------------------------------------

    let mut eps = SlepcEps::create(&world)?;
    eps.set_operators(Some(mat.as_raw()), None)?;
    eps.set_tolerances(Some(EPS_TOL), Some(EPS_MAXIT))?;
    eps.set_dimensions(Some(EPS_NEV), None, None)?;
    eps.set_which_eigenpairs(slepc_sys::EPSWhich::EPS_LARGEST_REAL)?;
    eps.set_type("krylovschur")?;
    eps.set_from_options()?;

    // ----------------------------------------------
    //             Solve the eigensystem
    // ----------------------------------------------
    eps.solve()?;

    // Get values and print
    let eps_its = eps.get_iteration_number()?;
    let eps_type = eps.get_type()?;
    let eps_nconv = eps.get_converged()?;
    let (eps_nev, _eps_ncv) = eps.get_dimensions()?;
    let (eps_tol, eps_maxit) = eps.get_tolerances()?;

    world.print(&format!(
        " Number of iterations of the method: {} \n",
        eps_its
    ))?;
    world.print(&format!(" Solution method: {} \n", eps_type))?;
    world.print(&format!(" Number of requested eigenvalues: {} \n", eps_nev))?;
    world.print(&format!(
        " Number of converged eigenvalues: {} \n",
        eps_nconv
    ))?;
    world.print(&format!(
        " Stopping condition: tol={:e}, maxit={} \n",
        eps_tol, eps_maxit,
    ))?;

    world.print(
        "\n           k          ||Ax-kx||/||kx||\n  ----------------- ------------------\n",
    )?;
    for i in 0..eps_nconv {
        let (kr, ki) = eps.get_eigenpair(i, xr.as_raw(), xi.as_raw())?;
        let error = eps.compute_error(i, slepc_sys::EPSErrorType::EPS_ERROR_RELATIVE)?;
        let (re, im) = (kr, ki);
        world.print(&format!("{:9.6} {:9.6}i {:12.2e} \n", re, im, error))?;
    }

    #[cfg(feature = "gnuplot")]
    {
        let (istart, iend) = xr.get_ownership_range()?;
        let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>())?;
        slepc_rs::gp::plot_line(&vec_vals);
    }

    Ok(())

    // mat.destroy();
    // x.destroy();
    // y.destroy();
    // xi.destroy();
    // xr.destroy();
    // eps.destroy();

    // SlepcWorld::finalize();
}
