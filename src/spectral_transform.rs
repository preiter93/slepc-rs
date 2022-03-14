//! Routines of `PETSc` spectral transform object [`slepc_sys::ST`]
//!
//! # Doc
//! <https://slepc.upv.es/documentation/slepc.pdf>
use crate::linear_system::PetscKSP;
use crate::vector::PetscVec;
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, Result};

pub struct PetscST {
    // Pointer to ST object
    pub st_p: *mut slepc_sys::_p_ST,
}

impl PetscST {
    /// Initialize from raw pointer
    pub fn from_raw(st_p: *mut slepc_sys::_p_ST) -> Self {
        Self { st_p }
    }

    // Return raw ST pointer
    pub fn as_raw(&self) -> *mut slepc_sys::_p_ST {
        self.st_p
    }

    /// Wrapper for [`slepc_sys::STCreate`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create(world: &SlepcWorld) -> Result<Self> {
        let (ierr, st_p) =
            unsafe { with_uninitialized(|st_p| slepc_sys::STCreate(world.as_raw(), st_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(st_p))
    }

    /// Wrapper for [`slepc_sys::STSetType`]
    ///
    /// Builds ST for a particular spectral transformation.
    ///
    /// ```text
    /// STSHELL       "shell"
    /// STSHIFT       "shift"
    /// STSINVERT     "sinvert"
    /// STCAYLEY      "cayley"
    /// STPRECOND     "precond"
    /// STFILTER      "filter"
    /// ```
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_type(&mut self, st_type: slepc_sys::STType) -> Result<()> {
        let ierr = unsafe { slepc_sys::STSetType(self.as_raw(), st_type) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::STSetShift`]
    ///
    /// Sets the shift associated with the spectral transformation.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_shift(&mut self, shift: slepc_sys::PetscScalar) -> Result<()> {
        let ierr = unsafe { slepc_sys::STSetShift(self.as_raw(), shift) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::STSetKSP`]
    ///
    /// Sets the KSP object associated with the spectral transformation.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_ksp(&mut self, ksp: &PetscKSP) -> Result<()> {
        let ierr = unsafe { slepc_sys::STSetKSP(self.as_raw(), ksp.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::STSetUp`]
    ///
    /// Prepares for the use of a spectral transformation.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_up(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::STSetUp(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::STApply`]
    ///
    /// Applies the spectral transformation operator to a vector,
    /// for instance  (A - sB)^-1 B in the case of the shift-and-invert
    /// transformation and generalized eigenproblem.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn apply(&mut self, x: &PetscVec, y: &mut PetscVec) -> Result<()> {
        let ierr = unsafe { slepc_sys::STApply(self.as_raw(), x.as_raw(), y.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::STGetType`]
    ///
    /// # Panics
    /// Casting `&str` to `CSring` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_type(&self) -> Result<&str> {
        let (ierr, st_type) =
            unsafe { with_uninitialized(|st_type| slepc_sys::STGetType(self.as_raw(), st_type)) };
        check_error(ierr)?;
        // Transform c string to rust string
        Ok(unsafe { std::ffi::CStr::from_ptr(st_type).to_str().unwrap() })
    }

    /// Wrapper for [`slepc_sys::STGetKSP`]
    ///
    /// Gets the KSP object associated with the spectral transformation.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_ksp(&self) -> Result<PetscKSP> {
        let (ierr, ksp) =
            unsafe { with_uninitialized(|ksp| slepc_sys::STGetKSP(self.as_raw(), ksp)) };
        check_error(ierr)?;
        Ok(PetscKSP::from_raw(ksp))
    }
}

impl Drop for PetscST {
    /// Wrapper for [`slepc_sys::STDestroy`]
    ///
    /// Frees space taken by a vector.
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::STDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from STDestroy", ierr);
        }
    }
}
