//! Routines of PETSc vectors [`slepc_sys::Vec`]
use crate::world::SlepcWorld;
use crate::{with_uninitialized, with_uninitialized2};

pub struct PetscVec {
    // // World communicator
    // pub world: &'a SlepcWorld,
    // Pointer to matrix object
    pub vec_p: *mut slepc_sys::_p_Vec,
}

impl PetscVec {
    /// Same as `Mat { ... }` but sets all optional params to `None`
    pub fn from_raw(vec_p: *mut slepc_sys::_p_Vec) -> Self {
        Self { vec_p }
    }

    /// Wrapper for [`slepc_sys::MatCreate`]
    pub fn create<'a>(world: &'a SlepcWorld) -> Self {
        let (ierr, vec_p) =
            unsafe { with_uninitialized(|vec_p| slepc_sys::VecCreate(world.as_raw(), vec_p)) };
        if ierr != 0 {
            println!("error code {} from VecCreate", ierr);
        }
        Self::from_raw(vec_p)
    }

    // Return raw `mat_p`
    pub fn as_raw(&self) -> *mut slepc_sys::_p_Vec {
        self.vec_p
    }

    /// Wrapper for [`slepc_sys::VecDuplicate`]
    pub fn duplicate(&self) -> Self {
        let (ierr, vec_p) =
            unsafe { with_uninitialized(|vec_p| slepc_sys::VecDuplicate(self.as_raw(), vec_p)) };
        if ierr != 0 {
            println!("error code {} from VecDuplicate", ierr);
        }
        Self::from_raw(vec_p)
    }

    /// Wrapper for [`slepc_sys::VecCopy`]
    ///
    /// Copy from x to Self
    pub fn copy(&mut self, x: &Self) {
        let ierr = unsafe { slepc_sys::VecCopy(x.as_raw(), self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from VecCopy", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecSetSizes`]
    pub fn set_sizes(
        &mut self,
        local_n: Option<slepc_sys::PetscInt>,
        global_n: Option<slepc_sys::PetscInt>,
    ) {
        let ierr = unsafe {
            slepc_sys::VecSetSizes(
                self.as_raw(),
                local_n.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
                global_n.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            )
        };
        if ierr != 0 {
            println!("error code {} from VecSetSizes", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecSetValues`]
    ///
    /// # Panics
    /// length mismatch of y and ix
    pub fn set_values(
        &mut self,
        ix: &[slepc_sys::PetscInt],
        y: &[slepc_sys::PetscScalar],
        iora: slepc_sys::InsertMode,
    ) {
        let ni = ix.len();
        assert_eq!(y.len(), ni);
        let ierr = unsafe {
            slepc_sys::VecSetValues(
                self.as_raw(),
                ni as slepc_sys::PetscInt,
                ix.as_ptr(),
                y.as_ptr() as *mut _,
                iora,
            )
        };
        if ierr != 0 {
            println!("error code {} from VecSetValues", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecSetFromOptions`]
    pub fn set_from_options(&mut self) {
        let ierr = unsafe { slepc_sys::VecSetFromOptions(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from VecSetFromOptions", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecSet`]
    ///
    /// Sets all components of a vector to a single scalar value `alpha`.
    pub fn set(&mut self, alpha: slepc_sys::PetscScalar) {
        let ierr = unsafe { slepc_sys::VecSet(self.as_raw(), alpha) };
        if ierr != 0 {
            println!("error code {} from VecSet", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecSetUp`]
    pub fn set_up(&mut self) {
        let ierr = unsafe { slepc_sys::VecSetUp(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from VecSetUp", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecAssemblyBegin`]
    pub fn assembly_begin(&self) {
        let ierr = unsafe { slepc_sys::VecAssemblyBegin(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from VecAssemblyBegin", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecAssemblyEnd`]
    pub fn assembly_end(&self) {
        let ierr = unsafe { slepc_sys::VecAssemblyEnd(self.as_raw()) };
        if ierr != 0 {
            println!("error code {} from VecAssemblyEnd", ierr);
        }
    }

    /// Wrapper for [`slepc_sys::VecGetValues`]
    ///
    /// # Example
    /// Get all vector values
    /// ```ignore
    /// let (istart, iend) = vec_p.get_ownership_range();
    /// let vec_vals = vec_p.get_values(&(istart..iend).collect::<Vec<i32>>());
    /// ```
    pub fn get_values(&self, ix: &[slepc_sys::PetscInt]) -> Vec<slepc_sys::PetscScalar> {
        let mut y = vec![slepc_sys::PetscScalar::default(); ix.len()];
        let ierr = unsafe {
            slepc_sys::VecGetValues(
                self.as_raw(),
                ix.len() as slepc_sys::PetscInt,
                ix.as_ptr(),
                y[..].as_mut_ptr() as *mut _,
            )
        };
        if ierr != 0 {
            println!("error code {} from VecGetValues", ierr);
        }
        y
    }

    /// Wrapper for [`slepc_sys::VecGetSize`]
    ///
    /// Returns the global number of elements of the vector.
    pub fn get_size(&self) -> slepc_sys::PetscInt {
        let (ierr, size) =
            unsafe { with_uninitialized(|size| slepc_sys::VecGetSize(self.as_raw(), size)) };
        if ierr != 0 {
            println!("error code {} from VecGetSize", ierr);
        }
        size
    }

    /// Wrapper for [`slepc_sys::VecGetLocalSize`]
    ///
    /// Returns the number of elements of the vector stored in local memory.
    pub fn get_local_size(&self) -> slepc_sys::PetscInt {
        let (ierr, size) =
            unsafe { with_uninitialized(|size| slepc_sys::VecGetLocalSize(self.as_raw(), size)) };
        if ierr != 0 {
            println!("error code {} from VecGetLocalSize", ierr);
        }
        size
    }

    /// Wrapper for [`slepc_sys::VecGetArrayRead`]
    pub fn get_array_read(&self) -> &[slepc_sys::PetscScalar] {
        let (ierr, x) =
            unsafe { with_uninitialized(|x| slepc_sys::VecGetArrayRead(self.as_raw(), x)) };
        if ierr != 0 {
            println!("error code {} from VecGetArrayRead", ierr);
        }
        // Get slice from raw pointer
        let size = self.get_local_size();
        unsafe { std::slice::from_raw_parts(x, size as usize) }
    }

    /// Wrapper for [`slepc_sys::VecGetArray`]
    pub fn get_array<'b>(&mut self) -> &'b mut [slepc_sys::PetscScalar] {
        let (ierr, x) = unsafe { with_uninitialized(|x| slepc_sys::VecGetArray(self.as_raw(), x)) };
        if ierr != 0 {
            println!("error code {} from VecGetArray", ierr);
        }
        // Get slice from raw pointer
        let size = self.get_local_size();
        unsafe { std::slice::from_raw_parts_mut(x, size as usize) }
    }

    /// Wrapper for [`slepc_sys::VecGetOwnershipRange`]
    /// Returns the range of matrix rows owned by this processor.
    pub fn get_ownership_range(&self) -> (slepc_sys::PetscInt, slepc_sys::PetscInt) {
        let (ierr, i_start, i_end) = unsafe {
            with_uninitialized2(|i_start, i_end| {
                slepc_sys::VecGetOwnershipRange(self.as_raw(), i_start, i_end)
            })
        };
        if ierr != 0 {
            println!("error code {} from VecGetOwnershipRange", ierr);
        }
        (i_start, i_end)
    }

    /// Wrapper for [`slepc_sys::VecDestroy`]
    pub fn destroy(&self) {
        let ierr = unsafe { slepc_sys::VecDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from VecDestroy", ierr);
        }
    }
}
