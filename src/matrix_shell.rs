//! Shell-Matrix routines of PETSc matrices
//!
//! Here used for matrix free eigenvalue problem.
use crate::matrix::PetscMat;
use crate::with_uninitialized;
use crate::world::SlepcWorld;

impl PetscMat {
    /// Wrapper for [`slepc_sys::MatCreateShell`]
    ///
    /// For matrix free eigenvalue problem.
    ///
    /// TODO: Figure out what the void pointer does ...
    pub fn create_shell<'a>(
        world: &'a SlepcWorld,
        local_rows: slepc_sys::PetscInt,
        local_cols: slepc_sys::PetscInt,
        global_rows: Option<slepc_sys::PetscInt>,
        global_cols: Option<slepc_sys::PetscInt>,
    ) -> Self {
        let (ierr, mat_p) = unsafe {
            with_uninitialized(|mat_p| {
                slepc_sys::MatCreateShell(
                    world.as_raw(),
                    local_rows,
                    local_cols,
                    global_rows.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    global_cols.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    std::ptr::null_mut(),
                    mat_p,
                )
            })
        };
        if ierr != 0 {
            println!("error code {} from MatCreateShell", ierr);
        }
        Self::from_raw(mat_p)
    }

    /// Wrapper for [`slepc_sys::MatShellSetOperation`]
    ///
    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// We split the `set_operation` into several functions, which must
    /// be chosen depending on the operation signature
    ///
    /// Type A  : fn(Mat, Vec) -> PetscErrorCode
    /// Type B  : fn(Mat, Vec, Vec) -> PetscErrorCode
    ///
    /// A used for: `MATOP_GET_DIAGONAL`
    /// B used for: `MATOP_MULT` `MATOP_MULT_TRANSPOSE`
    pub fn shell_set_operation_type_a(
        &self,
        op: slepc_sys::MatOperation,
        g: unsafe extern "C" fn(slepc_sys::Mat, slepc_sys::Vec) -> slepc_sys::PetscErrorCode,
    ) {
        match op {
            slepc_sys::MatOperation::MATOP_GET_DIAGONAL => (),
            // TODO: Add operations with same signature
            _ => panic!("The op: `{:?}` is not supported by operation_type_a", op),
        }
        let ierr = unsafe {
            slepc_sys::MatShellSetOperation(self.as_raw(), op, std::mem::transmute(Some(g)))
        };
        if ierr != 0 {
            println!("error code {} from MatShellSetOperation (Type A)", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatShellSetOperation`]
    ///
    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// We split the `set_operation` into several functions, which must
    /// be chosen depending on the operation signature
    ///
    /// Type A  : fn(Mat, Vec) -> PetscErrorCode
    /// Type B  : fn(Mat, Vec, Vec) -> PetscErrorCode
    ///
    /// A used for: `MATOP_GET_DIAGONAL`
    /// B used for: `MATOP_MULT` `MATOP_MULT_TRANSPOSE`
    pub fn shell_set_operation_type_b(
        &self,
        op: slepc_sys::MatOperation,
        g: unsafe extern "C" fn(
            slepc_sys::Mat,
            slepc_sys::Vec,
            slepc_sys::Vec,
        ) -> slepc_sys::PetscErrorCode,
    ) {
        match op {
            slepc_sys::MatOperation::MATOP_MULT | slepc_sys::MatOperation::MATOP_MULT_TRANSPOSE => {
                ()
            }
            // TODO: Add operations with same signature
            _ => panic!("The op: `{:?}` is not supported by operation_type_b", op),
        }
        let ierr = unsafe {
            slepc_sys::MatShellSetOperation(self.as_raw(), op, std::mem::transmute(Some(g)))
        };
        if ierr != 0 {
            println!("error code {} from MatShellSetOperation (Type B)", ierr);
        }
    }
}

/// Used for [`PetscMat::shell_set_operation_type_a`]
///
/// PETSc demands an unsafe C function with the signature
///
///     fn(Mat, Vec) -> PetscErrorCode
///
/// but our functions will have the signature
///
///     fn(PetscMat, PetscVec)
///
/// This trampoline macro creates the unsafe C function from
/// a user defined function. (can also be done manually)
///
/// See [`trampoline_type_b`] for example
#[macro_export]
macro_rules! trampoline_type_a {
    (
        $r: ident, $c: ident
    ) => {
        pub unsafe extern "C" fn $c(
            mat: slepc_sys::Mat,
            x: slepc_sys::Vec,
        ) -> slepc_sys::PetscErrorCode {
            let r_mat = PetscMat::from_raw(mat);
            let mut r_x = PetscVec::from_raw(x);
            $r(&r_mat, &mut r_x);
            0
        }
    };
}

/// Used for [`PetscMat::shell_set_operation_type_b`]
///
/// PETSc demands an unsafe C function with the signature
///
///     fn(Mat, Vec, Vec) -> PetscErrorCode
///
/// but our functions will have the signature
///
///     fn(PetscMat, PetscVec, PetscVec)
///
/// This trampoline macro creates the unsafe C function from
/// a user defined function. (can also be done manually)
///
/// # Use
/// First define a `my_mat_mult` function. For example
///``` ignore
/// fn my_mat_mult(_mat: &PetscMat, x: &PetscVec, y: &mut PetscVec) {
///    let x_view = x.get_array_read();
///    let y_view_mut = y.get_array();
///    let (i_start, i_end) = x.get_ownership_range();
///    let n = i_end - i_start;
///    for (i, y_i) in y_view_mut.iter_mut().enumerate() {
///        if i == 0 {
///            *y_i = 2. * x_view[i] - 1. * x_view[i + 1];
///        } else if i == n as usize - 1 {
///            *y_i = -1. * x_view[i - 1] + 2. * x_view[i];
///        } else {
///            *y_i = -1. * x_view[i - 1] + 2. * x_view[i] - 1. * x_view[i + 1];
///        }
///    }
/// }
/// ```
/// Then call this macro before `shell_set_operation_type_b`, i.e.
/// ``` ignore
/// slepc_rs::matrix_shell::trampoline_type_b!(my_mat_mult, my_mat_mult_raw);
/// mat.shell_set_operation_type_b(slepc_sys::MatOperation::MATOP_MULT, my_mat_mult_raw);
/// ```
#[macro_export]
macro_rules! trampoline_type_b {
    (
        $r: ident, $c: ident
    ) => {
        pub unsafe extern "C" fn $c(
            mat: slepc_sys::Mat,
            x: slepc_sys::Vec,
            y: slepc_sys::Vec,
        ) -> slepc_sys::PetscErrorCode {
            let r_mat = PetscMat::from_raw(mat);
            let r_x = PetscVec::from_raw(x);
            let mut r_y = PetscVec::from_raw(y);
            $r(&r_mat, &r_x, &mut r_y);
            0
        }
    };
}

pub use trampoline_type_b;
