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
        let a_mat_unwrap = if let Some(x) = a_mat {
            x.as_raw()
        } else {
            std::ptr::null_mut()
        };
        let p_mat_unwrap = if let Some(x) = p_mat {
            x.as_raw()
        } else {
            std::ptr::null_mut()
        };
        let ierr = unsafe { slepc_sys::PCSetOperators(self.as_raw(), a_mat_unwrap, p_mat_unwrap) };
        if ierr != 0 {
            println!("error code {} from PCSetOperators", ierr);
        }
    }
}
