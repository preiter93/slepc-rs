//! Preconditioner routines of `PETSc` preconditioners [`slepc_sys::PC`]
use crate::mat::PetscMat;
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, with_uninitialized2, Result};

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
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create(world: &SlepcWorld) -> Result<Self> {
        let (ierr, pc_p) =
            unsafe { with_uninitialized(|pc_p| slepc_sys::PCCreate(world.as_raw(), pc_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(pc_p))
    }

    /// Wrapper for [`slepc_sys::PCSetType`]
    ///
    /// Builds `PC` for a particular preconditioner type
    ///
    /// <https://petsc.org/main/docs/manualpages/PC/PCType.html#PCType>
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_type(&mut self, pc_type: &str) -> Result<()> {
        let pc_type_c = std::ffi::CString::new(pc_type)
            .expect("CString::new failed in preconditioner::set_type");
        let ierr = unsafe { slepc_sys::PCSetType(self.as_raw(), pc_type_c.as_ptr()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::PCSetOperators`]
    ///
    /// Sets the matrix associated with the linear system and a
    /// (possibly) different one associated with the preconditioner.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_operators(
        &mut self,
        a_mat: Option<PetscMat>,
        p_mat: Option<PetscMat>,
    ) -> Result<()> {
        let ierr = unsafe {
            slepc_sys::PCSetOperators(
                self.as_raw(),
                a_mat.map_or(std::ptr::null_mut(), |x| x.as_raw()),
                p_mat.map_or(std::ptr::null_mut(), |x| x.as_raw()),
            )
        };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::PCSetFromOptions`]
    ///
    /// Sets `PC` options from the options database. This routine must be called before
    /// `set_up` if the user is to be allowed to set the preconditioner method.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_from_options(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::PCSetFromOptions(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::PCSetUp`]
    ///
    /// Prepares for the use of a preconditioner.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_up(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::PCSetUp(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::PCGetOperators`]
    ///
    /// Gets the matrix associated with the linear system and
    /// possibly a different one associated with the preconditioner.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_operators(&mut self) -> Result<(PetscMat, PetscMat)> {
        let (ierr, a_mat, p_mat) = unsafe {
            with_uninitialized2(|a_mat, p_mat| {
                slepc_sys::PCGetOperators(self.as_raw(), a_mat, p_mat)
            })
        };
        check_error(ierr)?;
        Ok((PetscMat::from_raw(a_mat), PetscMat::from_raw(p_mat)))
    }
}

impl Drop for PetscPC {
    /// Wrapper for [`slepc_sys::PCDestroy`]
    ///
    /// Frees space taken by a preconditioner object.
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::PCDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from PCDestroy", ierr);
        }
    }
}
