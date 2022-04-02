//! Routines of `PETSc` viewer object [`slepc_sys::PetscViewer`]
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, Result};

pub struct PetscViewer {
    // Pointer to KSP object
    pub viewer_p: *mut slepc_sys::_p_PetscViewer,
}

impl PetscViewer {
    /// Initialize from raw pointer
    pub fn from_raw(viewer_p: *mut slepc_sys::_p_PetscViewer) -> Self {
        Self { viewer_p }
    }

    /// Return raw pointer
    pub fn as_raw(&self) -> *mut slepc_sys::_p_PetscViewer {
        self.viewer_p
    }

    /// Wrapper for [`slepc_sys::PetscViewerCreate`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create(world: &SlepcWorld) -> Result<Self> {
        let (ierr, viewer_p) = unsafe {
            with_uninitialized(|viewer_p| slepc_sys::PetscViewerCreate(world.as_raw(), viewer_p))
        };
        check_error(ierr)?;
        Ok(Self::from_raw(viewer_p))
    }

    /// Wrapper for [`slepc_sys::PETSC_VIEWER_STDOUT_`]
    pub fn stdout(comm: slepc_sys::MPI_Comm) -> Self {
        let viewer_p = unsafe { slepc_sys::PETSC_VIEWER_STDOUT_(comm) };
        Self::from_raw(viewer_p)
    }

    /// Wrapper for [`slepc_sys::PETSC_VIEWER_STDOUT_`]
    pub fn stdout_world() -> Self {
        Self::stdout(unsafe { slepc_sys::PETSC_COMM_WORLD })
    }
}

impl Drop for PetscViewer {
    /// Wrapper for [`slepc_sys::PetscViewerDestroy`]
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::PetscViewerDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from PetscViewerDestroy", ierr);
        }
    }
}
