//! Routines of `PETSc` vectors [`slepc_sys::Vec`]
use crate::world::SlepcWorld;
use crate::{check_error, with_uninitialized, with_uninitialized2, Result};

pub struct PetscVec {
    // Pointer to vector object
    pub vec_p: *mut slepc_sys::_p_Vec,
}

impl PetscVec {
    /// Initialize from raw pointer
    pub fn from_raw(vec_p: *mut slepc_sys::_p_Vec) -> Self {
        Self { vec_p }
    }

    /// Wrapper for [`slepc_sys::VecCreate`]
    ///
    /// Creates an empty vector object. The type can then be set with
    /// [`Self::set_type`], or [`Self::set_from_options`]..
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn create(world: &SlepcWorld) -> Result<Self> {
        let (ierr, vec_p) =
            unsafe { with_uninitialized(|vec_p| slepc_sys::VecCreate(world.as_raw(), vec_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(vec_p))
    }

    // Return raw `mat_p`
    pub fn as_raw(&self) -> *mut slepc_sys::_p_Vec {
        self.vec_p
    }

    /// Wrapper for [`slepc_sys::VecDuplicate`]
    ///
    /// Creates a new vector of the same type as an existing vector.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn duplicate(&self) -> Result<Self> {
        let (ierr, vec_p) =
            unsafe { with_uninitialized(|vec_p| slepc_sys::VecDuplicate(self.as_raw(), vec_p)) };
        check_error(ierr)?;
        Ok(Self::from_raw(vec_p))
    }

    /// Wrapper for [`slepc_sys::VecCopy`]
    ///
    /// Copy from x to Self
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn copy(&mut self, x: &Self) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecCopy(x.as_raw(), self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetSizes`]
    ///
    /// Sets the local and global sizes, and checks to determine compatibility.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_sizes(
        &mut self,
        local_n: Option<slepc_sys::PetscInt>,
        global_n: Option<slepc_sys::PetscInt>,
    ) -> Result<()> {
        let ierr = unsafe {
            slepc_sys::VecSetSizes(
                self.as_raw(),
                local_n.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
                global_n.unwrap_or(slepc_sys::PETSC_DECIDE_INTEGER),
            )
        };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetRandom`]
    ///
    /// Sets all components of a vector to random numbers.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_random(&mut self, rctx: Option<slepc_sys::PetscRandom>) -> Result<()> {
        let ierr =
            unsafe { slepc_sys::VecSetRandom(self.as_raw(), rctx.unwrap_or(std::ptr::null_mut())) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetValues`]
    ///
    ///  Inserts or adds values into certain locations of a vector.
    ///
    /// # Panics
    /// length mismatch of y and ix pr
    /// casting array size to `slepc_sys::PetscInt` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_values(
        &mut self,
        ix: &[slepc_sys::PetscInt],
        y: &[slepc_sys::PetscScalar],
        iora: slepc_sys::InsertMode,
    ) -> Result<()> {
        let ni = slepc_sys::PetscInt::try_from(ix.len()).unwrap();
        assert_eq!(y.len(), ix.len());
        let ierr = unsafe {
            slepc_sys::VecSetValues(self.as_raw(), ni, ix.as_ptr(), y.as_ptr() as *mut _, iora)
        };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetFromOptions`]
    ///
    /// Configures the vector from the options database.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_from_options(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecSetFromOptions(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetType`]
    ///
    /// Builds a vector, for a particular vector implementation.
    ///
    /// See `${PETSC_DIR}/include/petscvec.h` for available vector types
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_type(&mut self, vec_type: slepc_sys::VecType) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecSetType(self.as_raw(), vec_type) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSet`]
    ///
    /// Sets all components of a vector to a single scalar value `alpha`.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set(&mut self, alpha: slepc_sys::PetscScalar) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecSet(self.as_raw(), alpha) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecSetUp`]
    ///
    /// Sets up the internal vector data structures for the later use.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn set_up(&mut self) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecSetUp(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecAssemblyBegin`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn assembly_begin(&self) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecAssemblyBegin(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecAssemblyEnd`]
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn assembly_end(&self) -> Result<()> {
        let ierr = unsafe { slepc_sys::VecAssemblyEnd(self.as_raw()) };
        check_error(ierr)?;
        Ok(())
    }

    /// Wrapper for [`slepc_sys::VecGetValues`]
    ///
    /// # Example
    /// Get all vector values
    /// ```ignore
    /// let (istart, iend) = vec_p.get_ownership_range();
    /// let vec_vals = vec_p.get_values(&(istart..iend).collect::<Vec<i32>>());
    /// ```
    ///
    /// # Panics
    /// Casting array size to `slepc_sys::PetscInt` fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_values(&self, ix: &[slepc_sys::PetscInt]) -> Result<Vec<slepc_sys::PetscScalar>> {
        let mut y = vec![slepc_sys::PetscScalar::default(); ix.len()];
        let ni = slepc_sys::PetscInt::try_from(ix.len()).unwrap();
        let ierr = unsafe {
            slepc_sys::VecGetValues(self.as_raw(), ni, ix.as_ptr(), y[..].as_mut_ptr().cast())
        };
        check_error(ierr)?;
        Ok(y)
    }

    /// Wrapper for [`slepc_sys::VecGetSize`]
    ///
    /// Returns the global number of elements of the vector.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_size(&self) -> Result<slepc_sys::PetscInt> {
        let (ierr, size) =
            unsafe { with_uninitialized(|size| slepc_sys::VecGetSize(self.as_raw(), size)) };
        check_error(ierr)?;
        Ok(size)
    }

    /// Wrapper for [`slepc_sys::VecGetLocalSize`]
    ///
    /// Returns the number of elements of the vector stored in local memory.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_local_size(&self) -> Result<slepc_sys::PetscInt> {
        let (ierr, size) =
            unsafe { with_uninitialized(|size| slepc_sys::VecGetLocalSize(self.as_raw(), size)) };
        check_error(ierr)?;
        Ok(size)
    }

    /// Wrapper for [`slepc_sys::VecGetArrayRead`]
    ///
    /// Get read-only pointer to contiguous array containing this processor's portion of the vector data.
    ///
    /// # Panics
    /// Casting array size to usize fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_array_read<'a>(&'a self) -> Result<PetscVecView<'a>> {
        PetscVecView::from_vec(self)
    }

    /// Wrapper for [`slepc_sys::VecGetArray`]
    ///
    /// Returns a pointer to a contiguous array that contains this processor's portion of the vector data.
    ///
    /// # Panics
    /// Casting array size to usize fails
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_array<'a>(&'a mut self) -> Result<PetscVecViewMut<'a>> {
        PetscVecViewMut::from_vec(self)
    }

    /// Wrapper for [`slepc_sys::VecGetOwnershipRange`]
    ///
    /// Returns the range of matrix rows owned by this processor.
    ///
    /// # Errors
    /// `PETSc` returns error
    pub fn get_ownership_range(&self) -> Result<(slepc_sys::PetscInt, slepc_sys::PetscInt)> {
        let (ierr, i_start, i_end) = unsafe {
            with_uninitialized2(|i_start, i_end| {
                slepc_sys::VecGetOwnershipRange(self.as_raw(), i_start, i_end)
            })
        };
        check_error(ierr)?;
        Ok((i_start, i_end))
    }
}

impl Drop for PetscVec {
    /// Wrapper for [`slepc_sys::VecDestroy`]
    ///
    /// Frees space taken by a vector.
    fn drop(&mut self) {
        let ierr = unsafe { slepc_sys::VecDestroy(&mut self.as_raw() as *mut _) };
        if ierr != 0 {
            println!("error code {} from VecDestroy", ierr);
        }
    }
}

/// An immutable view, as returned from [`PetscVec::get_array_read`].
pub struct PetscVecView<'a> {
    pub(crate) vec: &'a PetscVec,
    pub(crate) array: *const slepc_sys::PetscScalar,
}

/// A mutable view, as returned from [`PetscVec::get_array`].
pub struct PetscVecViewMut<'a> {
    pub(crate) vec: &'a mut PetscVec,
    pub(crate) array: *mut slepc_sys::PetscScalar,
}


impl<'a> PetscVecView<'a> {
    /// Constructor
    fn from_vec(vec: &'a PetscVec) -> Result<Self> {
        let (ierr, array) =
            unsafe { with_uninitialized(|array| slepc_sys::VecGetArrayRead(vec.as_raw(), array)) };
        check_error(ierr)?;
        Ok(Self { vec, array })
    }

    /// Show as slice
    pub fn as_slice(&self) -> Result<&[slepc_sys::PetscScalar]> {
        // Get slice from raw pointer
        let size = usize::try_from(self.vec.get_local_size()?).unwrap();
        Ok(unsafe { std::slice::from_raw_parts(self.array, size) })
    }

    #[cfg(feature = "ndarray")]
    // Show as arrayview
    pub fn as_arrayview(&self) -> Result<ndarray_crate::ArrayView1<'a, slepc_sys::PetscScalar>> {
        // Get slice from raw pointer
        let size = usize::try_from(self.vec.get_local_size()?).unwrap();
        Ok(unsafe { ndarray_crate::ArrayView1::from_shape_ptr(size, self.array) })
    }
}

impl<'a> PetscVecViewMut<'a> {
    /// Constructor
    fn from_vec(vec: &'a mut PetscVec) -> Result<Self> {
        let (ierr, array) =
            unsafe { with_uninitialized(|array| slepc_sys::VecGetArray(vec.as_raw(), array)) };
        check_error(ierr)?;
        Ok(Self { vec, array })
    }

    /// Show as slice
    pub fn as_slice(&mut self) -> Result<&'a mut [slepc_sys::PetscScalar]> {
        // Get slice from raw pointer
        let size = usize::try_from(self.vec.get_local_size()?).unwrap();
        Ok(unsafe { std::slice::from_raw_parts_mut(self.array, size) })
    }

    #[cfg(feature = "ndarray")]
    // Show as arrayview
    pub fn as_arrayview(&mut self) -> Result<ndarray_crate::ArrayViewMut1<'a, slepc_sys::PetscScalar>> {
        // Get slice from raw pointer
        let size = usize::try_from(self.vec.get_local_size()?).unwrap();
        Ok(unsafe { ndarray_crate::ArrayViewMut1::from_shape_ptr(size, self.array) })
    }
}

impl Drop for PetscVecView<'_> {
    fn drop(&mut self) {
        let ierr = unsafe {
            slepc_sys::VecRestoreArrayRead(self.vec.vec_p, &mut self.array as *const _ as *mut _)
        };
        if ierr != 0 {
            println!("error code {} from VecRestoreArrayRead", ierr);
        }
    }
}

impl Drop for PetscVecViewMut<'_> {
    fn drop(&mut self) {
        let ierr = unsafe {
            slepc_sys::VecRestoreArray(self.vec.vec_p, &mut self.array as *mut _ as *mut _)
        };
        if ierr != 0 {
            println!("error code {} from VecRestoreArrad", ierr);
        }
    }
}
