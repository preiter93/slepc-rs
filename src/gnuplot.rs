#![cfg(feature = "with_gnuplot")]

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
pub fn plot_gnu(y: &[slepc_sys::PetscReal]) {
    use gnuplot::{Caption, Color, Figure};
    let x = (0..y.len())
        .map(|x| x as f64)
        .collect::<Vec<slepc_sys::PetscReal>>();
    let mut fg = Figure::new();
    fg.axes2d().lines(&x, y, &[Caption(""), Color("black")]);
    fg.show().unwrap();
}
