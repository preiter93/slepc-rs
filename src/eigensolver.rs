//! Eigenvalue solver of `SLEPc` [`slepc_sys::EPS`]
use crate::world::SlepcWorld;
use crate::{with_uninitialized, with_uninitialized2};

pub struct SlepcEps {
    // Pointer to EPS object
    pub eps: *mut slepc_sys::_p_EPS,
}

impl SlepcEps {
    /// Initialize from raw pointer
    fn from_raw(eps: *mut slepc_sys::_p_EPS) -> Self {
        Self { eps }
    }

    /// Wrapper for [`slepc_sys::EPSCreate`]
    pub fn create(world: &SlepcWorld) -> Self {
        let (ierr, eps) =
            unsafe { with_uninitialized(|eps| slepc_sys::EPSCreate(world.as_raw(), eps)) };
        if ierr != 0 {
            println!("error code {} from EPSCreate", ierr);
        }
        Self::from_raw(eps)
    }

    /// Return raw `eps`
    pub fn as_raw(&self) -> *mut slepc_sys::_p_EPS {
        self.eps
    }

    /// Wrapper for [`slepc_sys::EPSSetOperators`]
    ///
    /// Sets the matrices associated with the eigenvalue problem.
    /// `EPSSetOperators(EPS eps,Mat A,Mat B)`
    pub fn set_operators(&self, mat_a: Option<slepc_sys::Mat>, mat_b: Option<slepc_sys::Mat>) {
        let ierr = unsafe {
            slepc_sys::EPSSetOperators(
                self.as_raw(),
                mat_a.unwrap_or(std::ptr::null_mut()),
                mat_b.unwrap_or(std::ptr::null_mut()),
            )
        };
        if ierr != 0 {
            println!("error code {} from EPSSetOperators", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSetTolerances`]
    ///
    /// Set specific solver options
    pub fn set_tolerances(
        &mut self,
        tol: Option<slepc_sys::PetscReal>,
        maxit: Option<slepc_sys::PetscInt>,
    ) {
        let ierr = unsafe {
            slepc_sys::EPSSetTolerances(
                self.as_raw(),
                tol.unwrap_or(slepc_sys::PETSC_DEFAULT_REAL),
                maxit.unwrap_or(slepc_sys::PETSC_DEFAULT_INTEGER),
            )
        };
        if ierr != 0 {
            println!("error code {} from EPSSetTolerances", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSetDimensions`]
    ///
    /// Sets the number of eigenvalues (nev) to compute and the dimension of the subspace (ncv).
    /// `EPSSetDimensions(EPS eps,PetscInt nev,PetscInt ncv,PetscInt mpd)`
    ///
    /// Better to leave ncv and mpd untouched
    pub fn set_dimensions(
        &mut self,
        nev: Option<slepc_sys::PetscInt>,
        ncv: Option<slepc_sys::PetscInt>,
        mpd: Option<slepc_sys::PetscInt>,
    ) {
        let ierr = unsafe {
            slepc_sys::EPSSetDimensions(
                self.as_raw(),
                nev.unwrap_or(slepc_sys::PETSC_DEFAULT_INTEGER),
                ncv.unwrap_or(slepc_sys::PETSC_DEFAULT_INTEGER),
                mpd.unwrap_or(slepc_sys::PETSC_DEFAULT_INTEGER),
            )
        };
        if ierr != 0 {
            println!("error code {} from EPSSetDimensions", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSetFromOptions`]
    pub fn set_from_options(&self) {
        let ierr = unsafe { slepc_sys::EPSSetFromOptions(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from EPSSetFromOptions", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSetType`]
    ///
    /// The parameter 'which' can have one of these values
    /// ```text
    /// EPSPOWER       "power"
    /// EPSSUBSPACE    "subspace"
    /// EPSARNOLDI     "arnoldi"
    /// EPSLANCZOS     "lanczos"
    /// EPSKRYLOVSCHUR "krylovschur"
    /// ```
    /// More solvers:
    /// <https://slepc.upv.es/documentation/current/docs/manualpages/sys/EPSType.html#EPSType>
    pub fn set_type(&self, eps_type: &str) {
        let eps_type_c =
            std::ffi::CString::new(eps_type).expect("CString::new failed in eigensolver::set_type");
        let ierr = unsafe { slepc_sys::EPSSetType(self.as_raw(), eps_type_c.as_ptr()) };
        if ierr != 0 {
            println!("error code {} from EPSSetType", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSetWhichEigenpairs`]
    ///
    /// The parameter 'which' can have one of these values
    /// ```text
    /// EPS_LARGEST_MAGNITUDE    - largest eigenvalues in magnitude (default)
    /// EPS_SMALLEST_MAGNITUDE    - smallest eigenvalues in magnitude
    /// EPS_LARGEST_REAL    - largest real parts
    /// EPS_SMALLEST_REAL    - smallest real parts
    /// EPS_LARGEST_IMAGINARY    - largest imaginary parts
    /// EPS_SMALLEST_IMAGINARY     smallest imaginary parts
    /// EPS_TARGET_MAGNITUDE    - eigenvalues closest to the target (in magnitude)
    /// EPS_TARGET_REAL    - eigenvalues with real part closest to target
    /// EPS_TARGET_IMAGINARY    - eigenvalues with imaginary part closest to target
    /// EPS_ALL    - all eigenvalues contained in a given interval or region
    /// EPS_WHICH_USER    - user defined ordering set with EPSSetEigenvalueComparison()
    /// ```
    pub fn set_which_eigenpairs(&self, which: slepc_sys::EPSWhich) {
        let ierr = unsafe { slepc_sys::EPSSetWhichEigenpairs(self.as_raw(), which) };
        if ierr != 0 {
            println!("error code {} from EPSSetWhichEigenpairs", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSSolve`]
    ///
    /// Solve the eigensystem
    pub fn solve(&self) {
        let ierr = unsafe { slepc_sys::EPSSolve(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from EPSSolve", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::EPSGetTolerances`]
    pub fn get_tolerances(&self) -> (slepc_sys::PetscReal, slepc_sys::PetscInt) {
        let (ierr, tol, maxit) = unsafe {
            with_uninitialized2(|tol, maxit| slepc_sys::EPSGetTolerances(self.as_raw(), tol, maxit))
        };
        if ierr != 0 {
            println!("error code {} from EPSGetTolerances", ierr);
        }
        (tol, maxit)
    }

    /// Wrapper for [`slepc_sys::EPSGetIterationNumber`]
    pub fn get_iteration_number(&self) -> slepc_sys::PetscInt {
        let (ierr, its) = unsafe {
            with_uninitialized(|its| slepc_sys::EPSGetIterationNumber(self.as_raw(), its))
        };
        if ierr != 0 {
            println!("error code {} from EPSGetTolerances", ierr);
        }
        its
    }

    /// Wrapper for [`slepc_sys::EPSGetType`]
    ///
    /// # Panics
    /// Casting `&str` to `CSring` fails
    pub fn get_type(&self) -> &str {
        let (ierr, eps_type) = unsafe {
            with_uninitialized(|eps_type| slepc_sys::EPSGetType(self.as_raw(), eps_type))
        };
        if ierr != 0 {
            println!("error code {} from EPSGetType", ierr);
        }
        // Transform c string to rust string
        unsafe { std::ffi::CStr::from_ptr(eps_type).to_str().unwrap() }
    }

    /// Wrapper for [`slepc_sys::EPSGetConverged`]
    pub fn get_converged(&self) -> slepc_sys::PetscInt {
        let (ierr, nconv) =
            unsafe { with_uninitialized(|nconv| slepc_sys::EPSGetConverged(self.as_raw(), nconv)) };
        if ierr != 0 {
            println!("error code {} from EPSGetConverged", ierr);
        }
        nconv
    }

    /// Wrapper for [`slepc_sys::EPSGetDimensions`]
    ///
    /// TODO: Return also mpd if necessary
    pub fn get_dimensions(&self) -> (slepc_sys::PetscInt, slepc_sys::PetscInt) {
        let (ierr, nev, ncv) = unsafe {
            with_uninitialized2(|nev, ncv| {
                slepc_sys::EPSGetDimensions(self.as_raw(), nev, ncv, std::ptr::null_mut())
            })
        };
        if ierr != 0 {
            println!("error code {} from EPSGetDimensions", ierr);
        }
        (nev, ncv)
    }

    /// Wrapper for [`slepc_sys::EPSGetEigenpair`]
    pub fn get_eigenpair(
        &self,
        i: slepc_sys::PetscInt,
        xr: slepc_sys::Vec,
        xi: slepc_sys::Vec,
    ) -> (slepc_sys::PetscScalar, slepc_sys::PetscScalar) {
        let (ierr, kr, ki) = unsafe {
            with_uninitialized2(|kr, ki| {
                slepc_sys::EPSGetEigenpair(self.as_raw(), i, kr, ki, xr, xi)
            })
        };
        if ierr != 0 {
            println!("error code {} from EPSGetEigenpair", ierr);
        }
        (kr, ki)
    }

    /// Wrapper for [`slepc_sys::EPSComputeError`]
    pub fn compute_error(
        &self,
        i: slepc_sys::PetscInt,
        error_type: slepc_sys::EPSErrorType,
    ) -> slepc_sys::PetscReal {
        let (ierr, error) = unsafe {
            with_uninitialized(|error| {
                slepc_sys::EPSComputeError(self.as_raw(), i, error_type, error)
            })
        };
        if ierr != 0 {
            println!("error code {} from EPSComputeError", ierr);
        }
        error
    }

    /// Wrapper for [`slepc_sys::EPSDestroy`]
    pub fn destroy(&self) {
        let ierr = unsafe { slepc_sys::EPSDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from EPSDestroy", ierr);
        }
    }
}
