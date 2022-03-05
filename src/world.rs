//! Define slepc world
#![allow(clippy::module_name_repetitions)]
use crate::with_uninitialized;
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
    #[allow(clippy::cast_possible_truncation)]
    pub fn initialize() -> Self {
        // Command line arguments
        let argv = std::env::args().collect::<Vec<String>>();
        let argc = argv.len();
        let mut c_argc = argc as c_int;
        let mut c_argv = argv
            .into_iter()
            .map(|arg| CString::new(arg).unwrap().into_raw())
            .collect::<Vec<*mut c_char>>();
        let mut c_argv_ptr = c_argv.as_mut_ptr();

        // Initialize slepc
        let ierr = unsafe {
            slepc_sys::SlepcInitialize(
                &mut c_argc,
                &mut c_argv_ptr,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if ierr != 0 {
            println!("error code {} from SlepcInitialize", ierr);
        }

        Self(unsafe { slepc_sys::PETSC_COMM_WORLD })
    }

    // Finalize world
    pub fn finalize() {
        let ierr = unsafe { slepc_sys::SlepcFinalize() };
        if ierr != 0 {
            println!("error code {} from SlepcFinalize", ierr);
        }
    }

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
    pub fn print(&self, msg: &str) {
        let msg_c = CString::new(msg).unwrap();
        let ierr = unsafe { slepc_sys::PetscPrintf(self.as_raw(), msg_c.as_ptr()) };
        if ierr != 0 {
            println!("error code {} from PetscPrintf", ierr);
        }
    }
}
