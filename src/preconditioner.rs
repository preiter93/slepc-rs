//! Preconditioner routines of `PETSc` preconditioners [`slepc_sys::PC`]
use crate::matrix::PetscMat;
use crate::world::SlepcWorld;
use crate::{with_uninitialized, with_uninitialized2};

pub struct PetscPC {
    // Pointer to PC object
    pub pc_p: *mut slepc_sys::_p_PC,
}

impl PetscPC {
    /// Initialize from raw pointer
    pub fn from_raw(pc_p: *mut slepc_sys::_p_PC) -> Self {
        Self { pc_p }
    }

    // Return raw PC pointer
    pub fn as_raw(&self) -> *mut slepc_sys::_p_PC {
        self.pc_p
    }

    /// Wrapper for [`slepc_sys::PCCreate`]
    pub fn create(world: &SlepcWorld) -> Self {
        let (ierr, pc_p) =
            unsafe { with_uninitialized(|pc_p| slepc_sys::PCCreate(world.as_raw(), pc_p)) };
        if ierr != 0 {
            println!("error code {} from PCCreate", ierr);
        }
        Self::from_raw(pc_p)
    }

    /// Wrapper for [`slepc_sys::PCSetType`]
    ///
    /// Builds `PC` for a particular preconditioner type
    pub fn set_type(&mut self, pc_type: slepc_sys::PCType) {
        let ierr = unsafe { slepc_sys::PCSetType(self.as_raw(), pc_type) };
        if ierr != 0 {
            println!("error code {} from PCSetType", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::PCSetOperators`]
    ///
    /// Sets the matrix associated with the linear system and a
    /// (possibly) different one associated with the preconditioner.
    pub fn set_operators(&mut self, a_mat: Option<PetscMat>, p_mat: Option<PetscMat>) {
        let ierr = unsafe {
            slepc_sys::PCSetOperators(
                self.as_raw(),
                a_mat.map_or(std::ptr::null_mut(), |x| x.as_raw()),
                p_mat.map_or(std::ptr::null_mut(), |x| x.as_raw()),
            )
        };
        if ierr != 0 {
            println!("error code {} from PCSetOperators", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::PCSetFromOptions`]
    ///
    /// Sets `PC` options from the options database. This routine must be called before
    /// `set_up` if the user is to be allowed to set the preconditioner method.
    pub fn set_from_options(&mut self) {
        let ierr = unsafe { slepc_sys::PCSetFromOptions(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from PCSetFromOptions", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::PCSetUp`]
    ///
    /// Prepares for the use of a preconditioner.
    pub fn set_up(&mut self) {
        let ierr = unsafe { slepc_sys::PCSetUp(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from PCSetUp", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::PCGetOperators`]
    ///
    /// Gets the matrix associated with the linear system and
    /// possibly a different one associated with the preconditioner.
    pub fn get_operators(&mut self) -> (PetscMat, PetscMat) {
        let (ierr, a_mat, p_mat) = unsafe {
            with_uninitialized2(|a_mat, p_mat| {
                slepc_sys::PCGetOperators(self.as_raw(), a_mat, p_mat)
            })
        };
        if ierr != 0 {
            println!("error code {} from PCSetOperators", ierr);
        }
        (PetscMat::from_raw(a_mat), PetscMat::from_raw(p_mat))
    }

    /// Wrapper for [`slepc_sys::PCDestroy`]
    ///
    /// Frees space taken by a preconditioner object.
    pub fn destroy(&self) {
        let ierr = unsafe { slepc_sys::PCDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from PCDestroy", ierr);
        }
    }
}
