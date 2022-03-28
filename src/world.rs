//! Define slepc world
#![allow(clippy::module_name_repetitions)]
use crate::{check_error, with_uninitialized, Result};
use slepc_sys::MPI_Comm;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

/// Identifies a certain process within a communicator.
pub type Rank = c_int;

/// A user-defined communicator
pub struct SlepcWorld(MPI_Comm);

impl SlepcWorld {
    /// Initiailue slepc world
    ///
    /// # Panics
    /// Unwrap of `CString` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    #[allow(clippy::cast_possible_truncation)]
    pub fn initialize() -> Result<Self> {
        // Command line arguments
        let argv = std::env::args().collect::<Vec<String>>();
        let mut c_argv = argv
            .into_iter()
            .map(|arg| CString::new(arg).expect("CString::new failed").into_raw())
            .collect::<Vec<*mut c_char>>();
        // Without this, a segmentation fault occurs ...
        c_argv.push(std::ptr::null_mut());
        let c_argv_ptr = &mut c_argv.as_mut_ptr();
        let c_argc_ptr = &mut (c_argv.len() as c_int);

        // Initialize slepc
        let ierr = unsafe {
            slepc_sys::SlepcInitialize(c_argc_ptr, c_argv_ptr, std::ptr::null(), std::ptr::null())
        };
        check_error(ierr)?;
        Ok(Self(unsafe { slepc_sys::PETSC_COMM_WORLD }))
    }

    // // Finalize world
    // pub fn finalize() -> Result<()> {
    //     let ierr = unsafe { slepc_sys::SlepcFinalize() };
    //     check_error(ierr)?;
    //     Ok(())
    // }

    /// Whether the MPI library has been initialized
    pub fn is_initialized() -> bool {
        unsafe { with_uninitialized(|initialized| slepc_sys::MPI_Initialized(initialized)).1 != 0 }
    }

    // Return raw `Mpi_Comm`
    pub fn as_raw(&self) -> MPI_Comm {
        self.0
    }

    /// Number of processes in this communicator
    pub fn size(&self) -> Rank {
        unsafe { with_uninitialized(|size| slepc_sys::MPI_Comm_size(self.as_raw(), size)).1 }
    }

    /// The `Rank` that identifies the calling process within this communicator
    pub fn rank(&self) -> Rank {
        unsafe { with_uninitialized(|rank| slepc_sys::MPI_Comm_rank(self.as_raw(), rank)).1 }
    }

    /// Wrapper for [`slepc_sys::PetscPrintf`]
    ///
    /// Print a string
    ///
    /// # Panics
    /// Unwrap of `CString` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn print(&self, msg: &str) -> Result<()> {
        let msg_c = CString::new(msg).unwrap();
        let ierr = unsafe { slepc_sys::PetscPrintf(self.as_raw(), msg_c.as_ptr()) };
        check_error(ierr)?;
        Ok(())
    }
}

impl Drop for SlepcWorld {
    /// Finalize world
    ///
    /// TODO: Check if necessary
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::SlepcFinalize() };
        if ierr != 0 {
            println!("error code {} from SlepcFinalize", ierr);
        }
    }
}
