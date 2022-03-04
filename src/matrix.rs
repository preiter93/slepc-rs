//! Matrix routines of PETSc matrices [`slepc_sys::Mat`]
use crate::vector::PetscVec;
use crate::world::SlepcWorld;
use crate::{with_uninitialized, with_uninitialized2};

pub struct PetscMat {
    // // World communicator
    // pub world: &'a SlepcWorld,
    // Pointer to matrix object
    pub mat_p: *mut slepc_sys::_p_Mat,
}

impl PetscMat {
    /// Same as `Mat { ... }` but sets all optional params to `None`
    pub fn from_raw(mat_p: *mut slepc_sys::_p_Mat) -> Self {
        Self { mat_p }
    }

    /// Wrapper for [`slepc_sys::MatCreate`]
    pub fn create<'a>(world: &'a SlepcWorld) -> Self {
        let (ierr, mat_p) =
            unsafe { with_uninitialized(|mat_p| slepc_sys::MatCreate(world.as_raw(), mat_p)) };
        if ierr != 0 {
            println!("error code {} from MatCreate", ierr);
        }
        Self::from_raw(mat_p)
    }

    // Return raw `mat_p`
    pub fn as_raw(&self) -> *mut slepc_sys::_p_Mat {
        self.mat_p
    }

    /// Wrapper for [`slepc_sys::MatSetSizes`]
    ///
    /// TODO: Let error ...
    pub fn set_sizes(
        &mut self,
        local_rows: Option<slepc_sys::PetscInt>,
        local_cols: Option<slepc_sys::PetscInt>,
        global_rows: Option<slepc_sys::PetscInt>,
        global_cols: Option<slepc_sys::PetscInt>,
    ) {
        let ierr = unsafe {
            slepc_sys::MatSetSizes(
                self.as_raw(),
                local_rows.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
                local_cols.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
                global_rows.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
                global_cols.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            )
        };
        if ierr != 0 {
            println!("error code {} from MatSetSizes", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatSetValues`]
    /// By default the values, v, are row-oriented.
    ///
    /// # Panics
    /// length mismatch of v and n*m
    pub fn set_values(
        &mut self,
        idxm: &[slepc_sys::PetscInt],
        idxn: &[slepc_sys::PetscInt],
        v: &[slepc_sys::PetscScalar],
        addv: slepc_sys::InsertMode,
    ) {
        let m = idxm.len();
        let n = idxn.len();
        assert_eq!(v.len(), m * n);
        let ierr = unsafe {
            slepc_sys::MatSetValues(
                self.as_raw(),
                m as slepc_sys::PetscInt,
                idxm.as_ptr(),
                n as slepc_sys::PetscInt,
                idxn.as_ptr(),
                v.as_ptr() as *mut _,
                addv,
            )
        };
        if ierr != 0 {
            println!("error code {} from MatSetValues", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatSetFromOptions`]
    pub fn set_from_options(&mut self) {
        let ierr = unsafe { slepc_sys::MatSetFromOptions(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from MatSetFromOptions", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatSetUp`]
    pub fn set_up(&mut self) {
        let ierr = unsafe { slepc_sys::MatSetUp(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from MatSetUp", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatGetValues`]
    pub fn get_values(
        &self,
        idxm: &[slepc_sys::PetscInt],
        idxn: &[slepc_sys::PetscInt],
    ) -> Vec<slepc_sys::PetscScalar> {
        let mut v = vec![slepc_sys::PetscScalar::default(); idxm.len() * idxn.len()];
        let ierr = unsafe {
            slepc_sys::MatGetValues(
                self.as_raw(),
                idxm.len() as slepc_sys::PetscInt,
                idxm.as_ptr(),
                idxn.len() as slepc_sys::PetscInt,
                idxn.as_ptr(),
                v[..].as_mut_ptr() as *mut _,
            )
        };
        if ierr != 0 {
            println!("error code {} from MatGetValues", ierr);
        }
        v
    }

    /// Wrapper for [`slepc_sys::MatGetOwnershipRange`]
    /// Returns the range of matrix rows owned by this processor.
    pub fn get_ownership_range(&self) -> (slepc_sys::PetscInt, slepc_sys::PetscInt) {
        let (ierr, i_start, i_end) = unsafe {
            with_uninitialized2(|i_start, i_end| {
                slepc_sys::MatGetOwnershipRange(self.as_raw(), i_start, i_end)
            })
        };
        if ierr != 0 {
            println!("error code {} from MatGetOwnershipRange", ierr);
        }
        (i_start, i_end)
    }

    /// Wrapper for [`slepc_sys::MatAssemblyBegin`]
    pub fn assembly_begin(&self, assembly_type: slepc_sys::MatAssemblyType) {
        let ierr = unsafe { slepc_sys::MatAssemblyBegin(self.as_raw(), assembly_type) };
        if ierr != 0 {
            println!("error code {} from MatAssemblyBegin", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatAssemblyEnd`]
    pub fn assembly_end(&self, assembly_type: slepc_sys::MatAssemblyType) {
        let ierr = unsafe { slepc_sys::MatAssemblyEnd(self.as_raw(), assembly_type) };
        if ierr != 0 {
            println!("error code {} from MatAssemblyEnd", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::MatCreateVecs`]
    ///
    /// Return right - vector that the matrix can be multiplied against
    pub fn create_vec_right(&self) -> PetscVec {
        let (ierr, vec) = unsafe {
            with_uninitialized(|vec| {
                slepc_sys::MatCreateVecs(self.as_raw(), vec, std::ptr::null_mut())
            })
        };
        if ierr != 0 {
            println!("error code {} from MatCreateVecs", ierr);
        }
        PetscVec::from_raw(vec)
    }

    // / Wrapper for [`slepc_sys::MatCreateVecs`]
    // /
    // / Return left - vector that the matrix vector product can be stored in
    pub fn create_vec_left(&self) -> PetscVec {
        let (ierr, vec) = unsafe {
            with_uninitialized(|vec| {
                slepc_sys::MatCreateVecs(self.as_raw(), std::ptr::null_mut(), vec)
            })
        };
        if ierr != 0 {
            println!("error code {} from MatCreateVecs", ierr);
        }
        PetscVec::from_raw(vec)
    }
}
