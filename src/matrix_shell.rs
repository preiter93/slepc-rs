//! Shell-Matrix routines of `PETSc` matrices
//!
//! Here used for matrix free eigenvalue problem.
use crate::matrix::PetscMat;
use crate::vector::PetscVec;
use crate::with_uninitialized;
use crate::world::SlepcWorld;
use std::ffi::c_void;

impl PetscMat {
    /// Wrapper for [`slepc_sys::MatCreateShell`]
    ///
    /// For matrix free eigenvalue problem.
    ///
    /// Creates a new matrix class for use with a user-defined private
    ///  data storage format.
    ///
    /// Note:
    /// The context is necessary to retrieve information in operations
    /// like mat mul, about the size of the matrix etc. If no context is necessary,
    /// this can be set to None, such that a null pointer is used.
    /// However, in this case `create_shell`  requires an arbitraty type annotation,
    /// for example
    /// ```text
    /// let n = 100;
    /// let mat = PetscMat::create_shell::<u8>(&world, n, n, Some(n), Some(n), None);
    /// ```
    /// If you want to store a pointer to the size variable `n` as context to use
    /// later in other operations, you can do
    /// ```
    /// use slepc_rs::{world::SlepcWorld, matrix::PetscMat};
    /// // Initialize slepc
    /// let n = 100;
    /// let world = SlepcWorld::initialize();
    ///
    /// // Set context
    /// let mut ctx: (i32, i32) = (n, n);
    ///
    /// // Initialize shell matrix
    /// let mat = PetscMat::create_shell(&world, None, None, Some(n), Some(n), Some(&mut ctx));
    ///
    /// // Get context
    /// let ctx_: (i32, i32) = mat.shell_get_context().expect("No context found");
    ///
    /// assert_eq!(ctx, ctx_)
    /// ```
    pub fn create_shell<T>(
        world: &SlepcWorld,
        local_rows: Option<slepc_sys::PetscInt>,
        local_cols: Option<slepc_sys::PetscInt>,
        global_rows: Option<slepc_sys::PetscInt>,
        global_cols: Option<slepc_sys::PetscInt>,
        ctx: Option<&mut T>,
    ) -> Self {
        let ctx_void = ctx.map_or(std::ptr::null_mut(), |x| unsafe { ref_to_voidp(x) });
        let (ierr, mat_p) = unsafe {
            with_uninitialized(|mat_p| {
                slepc_sys::MatCreateShell(
                    world.as_raw(),
                    local_rows.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    local_cols.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    global_rows.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    global_cols.unwrap_or(slepc_sys::PETSC_DETERMINE_INTEGER),
                    ctx_void,
                    mat_p,
                )
            })
        };
        if ierr != 0 {
            println!("error code {} from MatCreateShell", ierr);
        }
        Self::from_raw(mat_p)
    }

    /// Wrapper for [`slepc_sys::MatShellGetContext`]
    ///
    /// Returns the user-provided context associated with a shell matrix.
    /// -> &'a mut T
    pub fn shell_get_context<T: std::marker::Copy>(&self) -> Option<T> {
        let mut ctx_void = std::mem::MaybeUninit::<*mut ::std::os::raw::c_void>::uninit();
        let ierr =
            unsafe { slepc_sys::MatShellGetContext(self.as_raw(), ctx_void.as_mut_ptr().cast()) };
        if ierr != 0 {
            println!("error code {} from MatShellGetContext", ierr);
        }
        let ctx_void = unsafe { ctx_void.assume_init() };
        // precautionary measure: ensure pointers are not null
        if ctx_void.is_null() {
            // println!(
            //     "\nAttention: Get context got a null pointer! {}\n{}\n",
            //     "The return values are most likely wrong.",
            //     "Make sure to have set a context in PetscMat::create_shell(..)"
            // );
            None
            // unsafe { *voidp_to_ref::<T>(&[].as_mut_ptr()) }
        } else {
            Some(unsafe { *voidp_to_ref(&ctx_void) })
        }
    }

    /// Wrapper for [`slepc_sys::MatShellSetOperation`]
    ///
    /// Allows user to set a matrix operation for a shell matrix.
    ///
    /// We split the `set_operation` into several functions, which must
    /// be chosen depending on the operation signature
    /// ```text
    /// Type A  : fn(&PetscMat, &mut PetscVec)
    /// Type B  : fn(&PetscMat, &PetscVec, &mut PetscVec)
    /// ```
    /// A used for: `MATOP_GET_DIAGONAL`
    /// B used for: `MATOP_MULT` `MATOP_MULT_TRANSPOSE`
    ///
    /// # Panics
    /// If passed `Matoperation` is not supported
    pub fn shell_set_operation_type_a<F: Fn(&PetscMat, &mut PetscVec)>(
        &self,
        op: slepc_sys::MatOperation,
        g: F,
    ) {
        match op {
            slepc_sys::MatOperation::MATOP_GET_DIAGONAL => (),
            // TODO: Add operations with same signature
            _ => panic!("The op: `{:?}` is not supported by operation_type_a", op),
        }
        let wrapped_g: unsafe extern "C" fn(
            slepc_sys::Mat,
            slepc_sys::Vec,
        ) -> slepc_sys::PetscErrorCode = wrap_callback_a(g);
        let ierr = unsafe {
            slepc_sys::MatShellSetOperation(self.as_raw(), op, std::mem::transmute(Some(wrapped_g)))
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
    /// ```text
    /// Type A  : fn(&PetscMat, &mut PetscVec)
    /// Type B  : fn(&PetscMat, &PetscVec, &mut PetscVec)
    /// ```
    /// A used for: `MATOP_GET_DIAGONAL`
    /// B used for: `MATOP_MULT` `MATOP_MULT_TRANSPOSE`
    ///
    /// # Panics
    /// If passed `Matoperation` is not supported
    pub fn shell_set_operation_type_b<F: Fn(&PetscMat, &PetscVec, &mut PetscVec)>(
        &self,
        op: slepc_sys::MatOperation,
        g: F,
    ) {
        match op {
            slepc_sys::MatOperation::MATOP_MULT | slepc_sys::MatOperation::MATOP_MULT_TRANSPOSE => {
            }
            // TODO: Add operations with same signature
            _ => panic!("The op: `{:?}` is not supported by operation_type_b", op),
        }
        let wrapped_g: unsafe extern "C" fn(
            slepc_sys::Mat,
            slepc_sys::Vec,
            slepc_sys::Vec,
        ) -> slepc_sys::PetscErrorCode = wrap_callback_b(g);

        let ierr = unsafe {
            slepc_sys::MatShellSetOperation(self.as_raw(), op, std::mem::transmute(Some(wrapped_g)))
        };
        if ierr != 0 {
            println!("error code {} from MatShellSetOperation (Type B)", ierr);
        }
    }
}

// /// Used for [`PetscMat::shell_set_operation_type_a`]
// ///
// /// `PETSc` demands an unsafe C function with the signature
// /// ```text
// ///     fn(Mat, Vec) -> PetscErrorCode
// /// ```
// /// but our functions will have the signature
// /// ```text
// ///     fn(PetscMat, PetscVec)
// /// ```
// /// This trampoline macro creates the unsafe C function from
// /// a user defined function. (can also be done manually)
// ///
// /// See [`trampoline_type_b`] for example
// #[macro_export]
// macro_rules! trampoline_type_a {
//     (
//         $r: ident, $c: ident
//     ) => {
//         pub unsafe extern "C" fn $c(
//             mat: $crate::slepc_sys::Mat,
//             x: $crate::slepc_sys::Vec,
//         ) -> $crate::slepc_sys::PetscErrorCode {
//             let r_mat = PetscMat::from_raw(mat);
//             let mut r_x = PetscVec::from_raw(x);
//             $r(&r_mat, &mut r_x);
//             // Avoid dropping
//             std::mem::forget(r_mat);
//             std::mem::forget(r_x);
//             0
//         }
//     };
// }

// /// Used for [`PetscMat::shell_set_operation_type_b`]
// ///
// /// `PETSc` demands an unsafe C function with the signature
// /// ```text
// ///     fn(Mat, Vec, Vec) -> PetscErrorCode
// /// ```
// /// but our functions will have the signature
// /// ```text
// ///     fn(PetscMat, PetscVec, PetscVec)
// /// ```
// /// This trampoline macro creates the unsafe C function from
// /// a user defined function. (can also be done manually)
// ///
// /// # Use
// /// First define a `my_mat_mult` function. For example
// ///``` ignore
// /// fn my_mat_mult(_mat: &PetscMat, x: &PetscVec, y: &mut PetscVec) {
// ///    let x_view = x.get_array_read();
// ///    let y_view_mut = y.get_array();
// ///    let (i_start, i_end) = x.get_ownership_range();
// ///    let n = i_end - i_start;
// ///    for (i, y_i) in y_view_mut.iter_mut().enumerate() {
// ///        if i == 0 {
// ///            *y_i = 2. * x_view[i] - 1. * x_view[i + 1];
// ///        } else if i == n as usize - 1 {
// ///            *y_i = -1. * x_view[i - 1] + 2. * x_view[i];
// ///        } else {
// ///            *y_i = -1. * x_view[i - 1] + 2. * x_view[i] - 1. * x_view[i + 1];
// ///        }
// ///    }
// /// }
// /// ```
// /// Then call this macro before `shell_set_operation_type_b`, i.e.
// /// ``` ignore
// /// slepc_rs::matrix_shell::trampoline_type_b!(my_mat_mult, my_mat_mult_raw);
// /// mat.shell_set_operation_type_b(slepc_sys::MatOperation::MATOP_MULT, my_mat_mult_raw);
// /// ```
// #[macro_export]
// macro_rules! trampoline_type_b {
//     (
//         $r: ident, $c: ident
//     ) => {
//         pub unsafe extern "C" fn $c(
//             mat: $crate::slepc_sys::Mat,
//             x: $crate::slepc_sys::Vec,
//             y: $crate::slepc_sys::Vec,
//         ) -> $crate::slepc_sys::PetscErrorCode {
//             let r_mat = PetscMat::from_raw(mat);
//             let r_x = PetscVec::from_raw(x);
//             let mut r_y = PetscVec::from_raw(y);
//             $r(&r_mat, &r_x, &mut r_y);
//             // Avoid dropping
//             std::mem::forget(r_mat);
//             std::mem::forget(r_x);
//             std::mem::forget(r_y);
//             0
//         }
//     };
// }

// pub use trampoline_type_a;
// pub use trampoline_type_b;

/// <https://users.rust-lang.org/t/converting-between-references-and-c-void/39599>
#[allow(dead_code)]
unsafe fn voidp_to_ref<'a, T: 'a>(&p: &'a *mut c_void) -> &'a T {
    &*p.cast()
}

/// <https://users.rust-lang.org/t/converting-between-references-and-c-void/39599>
#[allow(dead_code)]
unsafe fn ref_to_voidp<T>(r: &mut T) -> *mut c_void {
    (r as *mut T).cast::<std::ffi::c_void>()
}

/// Wrapping callbacks without userdata
/// <https://www.platymuus.com/posts/2016/callbacks-without-userdata/>
#[allow(clippy::items_after_statements)]
fn wrap_callback_a<F: Fn(&PetscMat, &mut PetscVec)>(
    _: F,
) -> unsafe extern "C" fn(slepc_sys::Mat, slepc_sys::Vec) -> slepc_sys::PetscErrorCode {
    assert!(std::mem::size_of::<F>() == 0);

    unsafe extern "C" fn wrapped<F: Fn(&PetscMat, &mut PetscVec)>(
        mat: slepc_sys::Mat,
        x: slepc_sys::Vec,
    ) -> slepc_sys::PetscErrorCode {
        let r_mat = std::mem::ManuallyDrop::new(PetscMat::from_raw(mat));
        let mut r_x = std::mem::ManuallyDrop::new(PetscVec::from_raw(x));
        // std::mem::transmute::<_, &F>(&())(&r_mat, &mut r_x);
        (&*(&() as *const ()).cast::<F>())(&r_mat, &mut r_x);
        0
    }
    wrapped::<F>
}

/// Wrapping callbacks without userdata
/// <https://www.platymuus.com/posts/2016/callbacks-without-userdata/>
#[allow(clippy::items_after_statements)]
fn wrap_callback_b<F: Fn(&PetscMat, &PetscVec, &mut PetscVec)>(
    _: F,
) -> unsafe extern "C" fn(slepc_sys::Mat, slepc_sys::Vec, slepc_sys::Vec) -> slepc_sys::PetscErrorCode
{
    assert!(std::mem::size_of::<F>() == 0);

    unsafe extern "C" fn wrapped<F: Fn(&PetscMat, &PetscVec, &mut PetscVec)>(
        mat: slepc_sys::Mat,
        x: slepc_sys::Vec,
        y: slepc_sys::Vec,
    ) -> slepc_sys::PetscErrorCode {
        let r_mat = std::mem::ManuallyDrop::new(PetscMat::from_raw(mat));
        let r_x = std::mem::ManuallyDrop::new(PetscVec::from_raw(x));
        let mut r_y = std::mem::ManuallyDrop::new(PetscVec::from_raw(y));
        // std::mem::transmute::<_, &F>(&())(&r_mat, &r_x, &mut r_y);
        (&*(&() as *const ()).cast::<F>())(&r_mat, &r_x, &mut r_y);
        0
    }
    wrapped::<F>
}
