//! Routines of `PETSc` options object [`slepc_sys::PetscOptions`]
use crate::{check_error, with_uninitialized, Result};

pub struct PetscOptions {
    // Pointer to KSP object
    pub options_p: *mut slepc_sys::_n_PetscOptions,
}

impl PetscOptions {
    /// Initialize from raw pointer
    pub fn from_raw(options_p: *mut slepc_sys::_n_PetscOptions) -> Self {
        Self { options_p }
    }

    /// Return raw pointer
    pub fn as_raw(&self) -> *mut slepc_sys::_n_PetscOptions {
        self.options_p
    }

    /// Wrapper for [`slepc_sys::PetscOptionsCreate`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create() -> Result<Self> {
        let (ierr, options_p) =
            unsafe { with_uninitialized(|options_p| slepc_sys::PetscOptionsCreate(options_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(options_p))
    }

    /// Wrapper for [`slepc_sys::PetscOptionsSetValue`]
    ///
    /// # Parameters
    /// - *options*: options database, use NULL for the default global database
    /// - *name*: name of option, this SHOULD have the - prepended
    /// - *value*: the option value (not used for all options, so can be NULL)
    ///
    /// # Example
    ///```ignore
    /// PetscOptions::set_value(None,"-st_ksp_converged_reason", None)?;
    ///```
    /// Same as command line argument.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_value(options: Option<&Self>, name: &str, value: Option<&str>) -> Result<()> {
        let name_c =
            std::ffi::CString::new(name).expect("CString::new failed in options::set_value");
        let value_c = std::ffi::CString::new(value.unwrap_or(""))
            .expect("CString::new failed in options::set_value");
        let ierr = match options {
            Some(x) => unsafe {
                slepc_sys::PetscOptionsSetValue(x.as_raw(), name_c.as_ptr(), value_c.as_ptr())
            },
            None => unsafe {
                slepc_sys::PetscOptionsSetValue(
                    std::ptr::null_mut(),
                    name_c.as_ptr(),
                    value_c.as_ptr(),
                )
            },
        };
        check_error(ierr)?;
        Ok(())
    }
}

impl Drop for PetscOptions {
    /// Wrapper for [`slepc_sys::PetscOptionsDestroy`]
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::PetscOptionsDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from PetscOptionsDestroy", ierr);
        }
    }
}
