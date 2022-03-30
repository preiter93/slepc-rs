//! Routines of `PETSc` linear system object [`slepc_sys::KSP`]
use crate::pc::PetscPC;
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, Result};
use std::mem::ManuallyDrop;

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

    /// Wrapper for [`slepc_sys::KSPSetFromOptions`]
    ///
    /// Sets `KSP` options from the options database. This routine must be called
    /// before `KSPSetUp()` if the user is to be allowed to set the Krylov type.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_from_options(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::KSPSetFromOptions(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
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
    pub fn set_type(&mut self, ksp_type: &str) -> Result<()> {
        let ksp_type_c = std::ffi::CString::new(ksp_type)
            .expect("CString::new failed in linear_system::set_type");
        let ierr = unsafe { slepc_sys::KSPSetType(self.as_raw(), ksp_type_c.as_ptr()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::KSPSetTolerances`]
    ///
    /// Sets the relative, absolute, divergence, and maximum iteration
    /// tolerances used by the default `KSP` convergence testers.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_tolerances(
        &mut self,
        rtol: Option<slepc_sys::PetscReal>,
        abstol: Option<slepc_sys::PetscReal>,
        dtol: Option<slepc_sys::PetscReal>,
        maxits: Option<slepc_sys::PetscInt>,
    ) -> Result<()> {
        let ierr = unsafe {
            slepc_sys::KSPSetTolerances(
                self.as_raw(),
                rtol.unwrap_or(slepc_sys::PETSC_DEFAULT_REAL),
                abstol.unwrap_or(slepc_sys::PETSC_DEFAULT_REAL),
                dtol.unwrap_or(slepc_sys::PETSC_DEFAULT_REAL),
                maxits.unwrap_or(slepc_sys::PETSC_DEFAULT_INTEGER),
            )
        };
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

    /// Wrapper for [`slepc_sys::KSPGetPC`]
    ///
    /// Returns a pointer to the preconditioner context set with ``KSPSetPC()``.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_pc(&self) -> Result<ManuallyDrop<PetscPC>> {
        let (ierr, pc) = unsafe { with_uninitialized(|pc| slepc_sys::KSPGetPC(self.as_raw(), pc)) };
        check_error(ierr)?;
        Ok(ManuallyDrop::new(PetscPC::from_raw(pc)))
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
