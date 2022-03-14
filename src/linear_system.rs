//! Routines of `PETSc` linear system object [`slepc_sys::KSP`]
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, Result};

pub struct PetscKSP {
    // Pointer to KSP object
    pub ksp_p: *mut slepc_sys::_p_KSP,
}

impl PetscKSP {
    /// Initialize from raw pointer
    pub fn from_raw(ksp_p: *mut slepc_sys::_p_KSP) -> Self {
        Self { ksp_p }
    }

    // Return raw ST pointer
    pub fn as_raw(&self) -> *mut slepc_sys::_p_KSP {
        self.ksp_p
    }

    /// Wrapper for [`slepc_sys::KSPCreate`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create(world: &SlepcWorld) -> Result<Self> {
        let (ierr, ksp_p) =
            unsafe { with_uninitialized(|ksp_p| slepc_sys::KSPCreate(world.as_raw(), ksp_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(ksp_p))
    }

    /// Wrapper for [`slepc_sys::KSPSetType`]
    ///
    /// Builds `PetscKSP` for a particular solver.
    ///
    /// All types:
    /// <https://petsc.org/main/docs/manualpages/KSP/KSPType.html#KSPType>
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_type(&mut self, ksp_type: slepc_sys::KSPType) -> Result<()> {
        let ierr = unsafe { slepc_sys::KSPSetType(self.as_raw(), ksp_type) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::KSPGetType`]
    ///
    /// # Panics
    /// Casting `&str` to `CSring` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_type(&self) -> Result<&str> {
        let (ierr, ksp_type) = unsafe {
            with_uninitialized(|ksp_type| slepc_sys::KSPGetType(self.as_raw(), ksp_type))
        };
        check_error(ierr)?;
        // Transform c string to rust string
        Ok(unsafe { std::ffi::CStr::from_ptr(ksp_type).to_str().unwrap() })
    }
}

impl Drop for PetscKSP {
    /// Wrapper for [`slepc_sys::KSPDestroy`]
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::KSPDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from KSPDestroy", ierr);
        }
    }
}
