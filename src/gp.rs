#![cfg(feature = "gnuplot")]
use gnuplot_crate::{Caption, Color, Figure};

/// Plot line
///
/// # Example
/// Plot Petsc Vector
/// ```ìgnore
/// let (istart, iend) = xr.get_ownership_range()?;
/// let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>())?;
/// plot_line(&vec_vals);
/// ```
///
/// # Panics
/// Gnuplot show fails.
#[allow(clippy::cast_precision_loss)]
pub fn plot_line(y: &[slepc_sys::PetscReal]) {
    let x = (0..y.len())
        .map(|x| x as f64)
        .collect::<Vec<slepc_sys::PetscReal>>();
    let mut fg = Figure::new();
    fg.axes2d().lines(&x, y, &[Caption(""), Color("black")]);
    fg.show().unwrap();
}

/// Plot line
///
/// # Example
/// Plot Petsc Vector
/// ```ìgnore
/// let (istart, iend) = xr.get_ownership_range();
/// let vec_vals = xr.get_values(&(istart..iend).collect::<Vec<i32>>());
/// plot_gnu(&vec_vals);
/// ```
///
/// # Panics
/// Gnuplot show fails.
#[allow(clippy::cast_precision_loss)]
pub fn plot_line2(x: &[slepc_sys::PetscReal], y: &[slepc_sys::PetscReal]) {
    assert!(x.len() == y.len(), "Size mismatch");
    let mut fg = Figure::new();
    fg.axes2d().lines(x, y, &[Caption(""), Color("black")]);
    fg.show().unwrap();
}

