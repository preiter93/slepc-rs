use num_complex::Complex;

/// use `num_complex::Complex` to allow for simpler arithmetic
pub type __BindgenComplex<T> = Complex<T>;
pub type PetscErrorCode = ::std::os::raw::c_int;
pub type PetscInt64 = i64;
pub type PetscInt = ::std::os::raw::c_int;
pub type PetscFloat = f32;
pub type PetscReal = f64;
pub type PetscComplex = __BindgenComplex<f64>;
#[cfg(feature = "scalar_complex")]
pub type PetscScalar = PetscComplex;
#[cfg(not(feature = "scalar_complex"))]
pub type PetscScalar = PetscReal;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PetscBool {
    PETSC_FALSE = 0,
    PETSC_TRUE = 1,
}

pub const PETSC_DECIDE: i32 = -1;
pub const PETSC_DETERMINE: i32 = -1;
pub const PETSC_DEFAULT: i32 = -2;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_PetscRandom {
    _unused: [u8; 0],
}
pub type PetscRandom = *mut _p_PetscRandom;

extern "C" {
    pub static mut PETSC_COMM_WORLD: MPI_Comm;
}

extern "C" {
    pub fn SlepcInitialize(
        arg1: *mut ::std::os::raw::c_int,
        arg2: *mut *mut *mut ::std::os::raw::c_char,
        arg3: *const ::std::os::raw::c_char,
        arg4: *const ::std::os::raw::c_char,
    ) -> PetscErrorCode;
}

extern "C" {
    pub fn SlepcFinalize() -> PetscErrorCode;
}

extern "C" {
    pub fn PetscPrintf(arg1: MPI_Comm, arg2: *const ::std::os::raw::c_char, ...) -> PetscErrorCode;
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum InsertMode {
    NOT_SET_VALUES = 0,
    INSERT_VALUES = 1,
    ADD_VALUES = 2,
    MAX_VALUES = 3,
    MIN_VALUES = 4,
    INSERT_ALL_VALUES = 5,
    ADD_ALL_VALUES = 6,
    INSERT_BC_VALUES = 7,
    ADD_BC_VALUES = 8,
}

// ---------- `Vec` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_Vec {
    _unused: [u8; 0],
}
pub type Vec = *mut _p_Vec;
pub type VecType = *const ::std::os::raw::c_char;
extern "C" {
    pub fn VecCreate(arg1: MPI_Comm, arg2: *mut Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecCopy(arg1: Vec, arg2: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecDuplicate(arg1: Vec, arg2: *mut Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSet(arg1: Vec, arg2: PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetUp(arg1: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetSizes(arg1: Vec, arg2: PetscInt, arg3: PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetType(arg1: Vec, arg2: VecType) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetRandom(arg1: Vec, arg2: PetscRandom) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetFromOptions(arg1: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecSetValues(
        arg1: Vec,
        arg2: PetscInt,
        arg3: *const PetscInt,
        arg4: *const PetscScalar,
        arg5: InsertMode,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetArray1d(
        arg1: Vec,
        arg2: PetscInt,
        arg3: PetscInt,
        arg4: *mut *mut PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetArray(arg1: Vec, arg2: *mut *mut PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetArrayWrite(arg1: Vec, arg2: *mut *mut PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetArrayRead(arg1: Vec, arg2: *mut *const PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetSize(arg1: Vec, arg2: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetLocalSize(arg1: Vec, arg2: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn VecAssemblyBegin(arg1: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecAssemblyEnd(arg1: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetValues(
        arg1: Vec,
        arg2: PetscInt,
        arg3: *const PetscInt,
        arg4: *mut PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetOwnershipRange(
        arg1: Vec,
        arg2: *mut PetscInt,
        arg3: *mut PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn VecGetOwnershipRanges(arg1: Vec, arg2: *mut *const PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn VecDestroy(arg1: *mut Vec) -> PetscErrorCode;
}

// ---------- `Mat` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_Mat {
    _unused: [u8; 0],
}
pub type Mat = *mut _p_Mat;
pub type MatType = *const ::std::os::raw::c_char;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MatAssemblyType {
    MAT_FLUSH_ASSEMBLY = 1,
    MAT_FINAL_ASSEMBLY = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MatOperation {
    MATOP_SET_VALUES = 0,
    MATOP_GET_ROW = 1,
    MATOP_RESTORE_ROW = 2,
    MATOP_MULT = 3,
    MATOP_MULT_ADD = 4,
    MATOP_MULT_TRANSPOSE = 5,
    MATOP_MULT_TRANSPOSE_ADD = 6,
    MATOP_SOLVE = 7,
    MATOP_SOLVE_ADD = 8,
    MATOP_SOLVE_TRANSPOSE = 9,
    MATOP_SOLVE_TRANSPOSE_ADD = 10,
    MATOP_LUFACTOR = 11,
    MATOP_CHOLESKYFACTOR = 12,
    MATOP_SOR = 13,
    MATOP_TRANSPOSE = 14,
    MATOP_GETINFO = 15,
    MATOP_EQUAL = 16,
    MATOP_GET_DIAGONAL = 17,
    MATOP_DIAGONAL_SCALE = 18,
    MATOP_NORM = 19,
    MATOP_ASSEMBLY_BEGIN = 20,
    MATOP_ASSEMBLY_END = 21,
    MATOP_SET_OPTION = 22,
    MATOP_ZERO_ENTRIES = 23,
    MATOP_ZERO_ROWS = 24,
    MATOP_LUFACTOR_SYMBOLIC = 25,
    MATOP_LUFACTOR_NUMERIC = 26,
    MATOP_CHOLESKY_FACTOR_SYMBOLIC = 27,
    MATOP_CHOLESKY_FACTOR_NUMERIC = 28,
    MATOP_SETUP_PREALLOCATION = 29,
    MATOP_ILUFACTOR_SYMBOLIC = 30,
    MATOP_ICCFACTOR_SYMBOLIC = 31,
    MATOP_GET_DIAGONAL_BLOCK = 32,
    MATOP_FREE_INTER_STRUCT = 33,
    MATOP_DUPLICATE = 34,
    MATOP_FORWARD_SOLVE = 35,
    MATOP_BACKWARD_SOLVE = 36,
    MATOP_ILUFACTOR = 37,
    MATOP_ICCFACTOR = 38,
    MATOP_AXPY = 39,
    MATOP_CREATE_SUBMATRICES = 40,
    MATOP_INCREASE_OVERLAP = 41,
    MATOP_GET_VALUES = 42,
    MATOP_COPY = 43,
    MATOP_GET_ROW_MAX = 44,
    MATOP_SCALE = 45,
    MATOP_SHIFT = 46,
    MATOP_DIAGONAL_SET = 47,
    MATOP_ZERO_ROWS_COLUMNS = 48,
    MATOP_SET_RANDOM = 49,
    MATOP_GET_ROW_IJ = 50,
    MATOP_RESTORE_ROW_IJ = 51,
    MATOP_GET_COLUMN_IJ = 52,
    MATOP_RESTORE_COLUMN_IJ = 53,
    MATOP_FDCOLORING_CREATE = 54,
    MATOP_COLORING_PATCH = 55,
    MATOP_SET_UNFACTORED = 56,
    MATOP_PERMUTE = 57,
    MATOP_SET_VALUES_BLOCKED = 58,
    MATOP_CREATE_SUBMATRIX = 59,
    MATOP_DESTROY = 60,
    MATOP_VIEW = 61,
    MATOP_CONVERT_FROM = 62,
    MATOP_MATMAT_MULT = 63,
    MATOP_MATMAT_MULT_SYMBOLIC = 64,
    MATOP_MATMAT_MULT_NUMERIC = 65,
    MATOP_SET_LOCAL_TO_GLOBAL_MAP = 66,
    MATOP_SET_VALUES_LOCAL = 67,
    MATOP_ZERO_ROWS_LOCAL = 68,
    MATOP_GET_ROW_MAX_ABS = 69,
    MATOP_GET_ROW_MIN_ABS = 70,
    MATOP_CONVERT = 71,
    MATOP_SET_COLORING = 72,
    MATOP_SET_VALUES_ADIFOR = 74,
    MATOP_FD_COLORING_APPLY = 75,
    MATOP_SET_FROM_OPTIONS = 76,
    MATOP_MULT_CONSTRAINED = 77,
    MATOP_MULT_TRANSPOSE_CONSTRAIN = 78,
    MATOP_FIND_ZERO_DIAGONALS = 79,
    MATOP_MULT_MULTIPLE = 80,
    MATOP_SOLVE_MULTIPLE = 81,
    MATOP_GET_INERTIA = 82,
    MATOP_LOAD = 83,
    MATOP_IS_SYMMETRIC = 84,
    MATOP_IS_HERMITIAN = 85,
    MATOP_IS_STRUCTURALLY_SYMMETRIC = 86,
    MATOP_SET_VALUES_BLOCKEDLOCAL = 87,
    MATOP_CREATE_VECS = 88,
    MATOP_MAT_MULT = 89,
    MATOP_MAT_MULT_SYMBOLIC = 90,
    MATOP_MAT_MULT_NUMERIC = 91,
    MATOP_PTAP = 92,
    MATOP_PTAP_SYMBOLIC = 93,
    MATOP_PTAP_NUMERIC = 94,
    MATOP_MAT_TRANSPOSE_MULT = 95,
    MATOP_MAT_TRANSPOSE_MULT_SYMBO = 96,
    MATOP_MAT_TRANSPOSE_MULT_NUMER = 97,
    MATOP_PRODUCTSETFROMOPTIONS = 99,
    MATOP_PRODUCTSYMBOLIC = 100,
    MATOP_PRODUCTNUMERIC = 101,
    MATOP_CONJUGATE = 102,
    MATOP_SET_VALUES_ROW = 104,
    MATOP_REAL_PART = 105,
    MATOP_IMAGINARY_PART = 106,
    MATOP_GET_ROW_UPPER_TRIANGULAR = 107,
    MATOP_RESTORE_ROW_UPPER_TRIANG = 108,
    MATOP_MAT_SOLVE = 109,
    MATOP_MAT_SOLVE_TRANSPOSE = 110,
    MATOP_GET_ROW_MIN = 111,
    MATOP_GET_COLUMN_VECTOR = 112,
    MATOP_MISSING_DIAGONAL = 113,
    MATOP_GET_SEQ_NONZERO_STRUCTUR = 114,
    MATOP_CREATE = 115,
    MATOP_GET_GHOSTS = 116,
    MATOP_GET_LOCAL_SUB_MATRIX = 117,
    MATOP_RESTORE_LOCALSUB_MATRIX = 118,
    MATOP_MULT_DIAGONAL_BLOCK = 119,
    MATOP_HERMITIAN_TRANSPOSE = 120,
    MATOP_MULT_HERMITIAN_TRANSPOSE = 121,
    MATOP_MULT_HERMITIAN_TRANS_ADD = 122,
    MATOP_GET_MULTI_PROC_BLOCK = 123,
    MATOP_FIND_NONZERO_ROWS = 124,
    MATOP_GET_COLUMN_NORMS = 125,
    MATOP_INVERT_BLOCK_DIAGONAL = 126,
    MATOP_CREATE_SUB_MATRICES_MPI = 128,
    MATOP_SET_VALUES_BATCH = 129,
    MATOP_TRANSPOSE_MAT_MULT = 130,
    MATOP_TRANSPOSE_MAT_MULT_SYMBO = 131,
    MATOP_TRANSPOSE_MAT_MULT_NUMER = 132,
    MATOP_TRANSPOSE_COLORING_CREAT = 133,
    MATOP_TRANS_COLORING_APPLY_SPT = 134,
    MATOP_TRANS_COLORING_APPLY_DEN = 135,
    MATOP_RART = 136,
    MATOP_RART_SYMBOLIC = 137,
    MATOP_RART_NUMERIC = 138,
    MATOP_SET_BLOCK_SIZES = 139,
    MATOP_AYPX = 140,
    MATOP_RESIDUAL = 141,
    MATOP_FDCOLORING_SETUP = 142,
    MATOP_MPICONCATENATESEQ = 144,
    MATOP_DESTROYSUBMATRICES = 145,
    MATOP_TRANSPOSE_SOLVE = 146,
    MATOP_GET_VALUES_LOCAL = 147,
}
extern "C" {
    pub fn MatDestroy(arg1: *mut Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn MatCreate(arg1: MPI_Comm, arg2: *mut Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn MatCreateShell(
        arg1: MPI_Comm,
        arg2: PetscInt,
        arg3: PetscInt,
        arg4: PetscInt,
        arg5: PetscInt,
        arg6: *mut ::std::os::raw::c_void,
        arg7: *mut Mat,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatCreateVecs(arg1: Mat, arg2: *mut Vec, arg3: *mut Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetSizes(
        arg1: Mat,
        arg2: PetscInt,
        arg3: PetscInt,
        arg4: PetscInt,
        arg5: PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetUp(arg1: Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetType(arg1: Mat, arg2: MatType) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetFromOptions(arg1: Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetValues(
        arg1: Mat,
        arg2: PetscInt,
        arg3: *const PetscInt,
        arg4: PetscInt,
        arg5: *const PetscInt,
        arg6: *const PetscScalar,
        arg7: InsertMode,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetValuesRow(arg1: Mat, arg2: PetscInt, arg3: *const PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetValuesRowLocal(
        arg1: Mat,
        arg2: PetscInt,
        arg3: *const PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatSetRandom(arg1: Mat, arg2: PetscRandom) -> PetscErrorCode;
}
extern "C" {
    pub fn MatShellSetOperation(
        arg1: Mat,
        arg2: MatOperation,
        arg3: ::std::option::Option<unsafe extern "C" fn()>,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatAssemblyBegin(arg1: Mat, arg2: MatAssemblyType) -> PetscErrorCode;
}
extern "C" {
    pub fn MatAssemblyEnd(arg1: Mat, arg2: MatAssemblyType) -> PetscErrorCode;
}
extern "C" {
    pub fn MatAssembled(arg1: Mat, arg2: *mut PetscBool) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetType(arg1: Mat, arg2: *mut MatType) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetValues(
        arg1: Mat,
        arg2: PetscInt,
        arg3: *const PetscInt,
        arg4: PetscInt,
        arg5: *const PetscInt,
        arg6: *mut PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetRow(
        arg1: Mat,
        arg2: PetscInt,
        arg3: *mut PetscInt,
        arg4: *mut *const PetscInt,
        arg5: *mut *const PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetSize(arg1: Mat, arg2: *mut PetscInt, arg3: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetLocalSize(arg1: Mat, arg2: *mut PetscInt, arg3: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetOwnershipRange(
        arg1: Mat,
        arg2: *mut PetscInt,
        arg3: *mut PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn MatGetOwnershipRanges(arg1: Mat, arg2: *mut *const PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn MatShellGetContext(arg1: Mat, arg2: *mut ::std::os::raw::c_void) -> PetscErrorCode;
}
// ---------- `PC` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_PC {
    _unused: [u8; 0],
}
pub type PC = *mut _p_PC;
pub type PCType = *const ::std::os::raw::c_char;
extern "C" {
    pub fn PCCreate(arg1: MPI_Comm, arg2: *mut PC) -> PetscErrorCode;
}
extern "C" {
    pub fn PCSetType(arg1: PC, arg2: PCType) -> PetscErrorCode;
}
extern "C" {
    pub fn PCGetType(arg1: PC, arg2: *mut PCType) -> PetscErrorCode;
}
extern "C" {
    pub fn PCSetUp(arg1: PC) -> PetscErrorCode;
}
extern "C" {
    pub fn PCReset(arg1: PC) -> PetscErrorCode;
}
extern "C" {
    pub fn PCDestroy(arg1: *mut PC) -> PetscErrorCode;
}
extern "C" {
    pub fn PCSetFromOptions(arg1: PC) -> PetscErrorCode;
}
extern "C" {
    pub fn PCSetOperators(arg1: PC, arg2: Mat, arg3: Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn PCGetOperators(arg1: PC, arg2: *mut Mat, arg3: *mut Mat) -> PetscErrorCode;
}
// ---------- `KSP` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_KSP {
    _unused: [u8; 0],
}
pub type KSP = *mut _p_KSP;
pub type KSPType = *const ::std::os::raw::c_char;
extern "C" {
    pub fn KSPCreate(arg1: MPI_Comm, arg2: *mut KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPDestroy(arg1: *mut KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPSetType(arg1: KSP, arg2: KSPType) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPGetType(arg1: KSP, arg2: *mut KSPType) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPSetUp(arg1: KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPSetFromOptions(arg1: KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPSetTolerances(
        arg1: KSP,
        arg2: PetscReal,
        arg3: PetscReal,
        arg4: PetscReal,
        arg5: PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPGetTolerances(
        arg1: KSP,
        arg2: *mut PetscReal,
        arg3: *mut PetscReal,
        arg4: *mut PetscReal,
        arg5: *mut PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn KSPGetPC(arg1: KSP, arg2: *mut PC) -> PetscErrorCode;
}
// ---------- `ST` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_ST {
    _unused: [u8; 0],
}
pub type ST = *mut _p_ST;
pub type STType = *const ::std::os::raw::c_char;
extern "C" {
    pub fn STCreate(arg1: MPI_Comm, arg2: *mut ST) -> PetscErrorCode;
}
extern "C" {
    pub fn STDestroy(arg1: *mut ST) -> PetscErrorCode;
}
extern "C" {
    pub fn STReset(arg1: ST) -> PetscErrorCode;
}
extern "C" {
    pub fn STSetType(arg1: ST, arg2: STType) -> PetscErrorCode;
}
extern "C" {
    pub fn STGetType(arg1: ST, arg2: *mut STType) -> PetscErrorCode;
}
extern "C" {
    pub fn STSetKSP(arg1: ST, arg2: KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn STGetKSP(arg1: ST, arg2: *mut KSP) -> PetscErrorCode;
}
extern "C" {
    pub fn STApply(arg1: ST, arg2: Vec, arg3: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn STSetUp(arg1: ST) -> PetscErrorCode;
}
extern "C" {
    pub fn STSetFromOptions(arg1: ST) -> PetscErrorCode;
}
extern "C" {
    pub fn STSetShift(arg1: ST, arg2: PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn STGetShift(arg1: ST, arg2: *mut PetscScalar) -> PetscErrorCode;
}
// ---------- `EPS` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _p_EPS {
    _unused: [u8; 0],
}
pub type EPS = *mut _p_EPS;
pub type EPSType = *const ::std::os::raw::c_char;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EPSWhich {
    EPS_LARGEST_MAGNITUDE = 1,
    EPS_SMALLEST_MAGNITUDE = 2,
    EPS_LARGEST_REAL = 3,
    EPS_SMALLEST_REAL = 4,
    EPS_LARGEST_IMAGINARY = 5,
    EPS_SMALLEST_IMAGINARY = 6,
    EPS_TARGET_MAGNITUDE = 7,
    EPS_TARGET_REAL = 8,
    EPS_TARGET_IMAGINARY = 9,
    EPS_ALL = 10,
    EPS_WHICH_USER = 11,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EPSErrorType {
    EPS_ERROR_ABSOLUTE = 0,
    EPS_ERROR_RELATIVE = 1,
    EPS_ERROR_BACKWARD = 2,
}
extern "C" {
    pub static mut EPSErrorTypes: [*const ::std::os::raw::c_char; 0usize];
}
extern "C" {
    pub fn EPSCreate(arg1: MPI_Comm, arg2: *mut EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSDestroy(arg1: *mut EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSReset(arg1: EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetType(arg1: EPS, arg2: EPSType) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetType(arg1: EPS, arg2: *mut EPSType) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetOperators(arg1: EPS, arg2: Mat, arg3: Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetOperators(arg1: EPS, arg2: *mut Mat, arg3: *mut Mat) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetFromOptions(arg1: EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetUp(arg1: EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSolve(arg1: EPS) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetTarget(arg1: EPS, arg2: PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetTarget(arg1: EPS, arg2: *mut PetscScalar) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetST(arg1: EPS, arg2: ST) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetST(arg1: EPS, arg2: *mut ST) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetTolerances(arg1: EPS, arg2: PetscReal, arg3: PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetTolerances(arg1: EPS, arg2: *mut PetscReal, arg3: *mut PetscInt)
        -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetDimensions(
        arg1: EPS,
        arg2: PetscInt,
        arg3: PetscInt,
        arg4: PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetDimensions(
        arg1: EPS,
        arg2: *mut PetscInt,
        arg3: *mut PetscInt,
        arg4: *mut PetscInt,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetConverged(arg1: EPS, arg2: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetEigenpair(
        arg1: EPS,
        arg2: PetscInt,
        arg3: *mut PetscScalar,
        arg4: *mut PetscScalar,
        arg5: Vec,
        arg6: Vec,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetEigenvalue(
        arg1: EPS,
        arg2: PetscInt,
        arg3: *mut PetscScalar,
        arg4: *mut PetscScalar,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetEigenvector(arg1: EPS, arg2: PetscInt, arg3: Vec, arg4: Vec) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetLeftEigenvector(arg1: EPS, arg2: PetscInt, arg3: Vec, arg4: Vec)
        -> PetscErrorCode;
}
extern "C" {
    pub fn EPSComputeError(
        arg1: EPS,
        arg2: PetscInt,
        arg3: EPSErrorType,
        arg4: *mut PetscReal,
    ) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetIterationNumber(arg1: EPS, arg2: *mut PetscInt) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSSetWhichEigenpairs(arg1: EPS, arg2: EPSWhich) -> PetscErrorCode;
}
extern "C" {
    pub fn EPSGetWhichEigenpairs(arg1: EPS, arg2: *mut EPSWhich) -> PetscErrorCode;
}
// ---------- `MPI` ----------
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ompi_communicator_t {
    _unused: [u8; 0],
}
pub type MPI_Comm = *mut ompi_communicator_t;

extern "C" {
    pub fn MPI_Initialized(flag: *mut ::std::os::raw::c_int) -> ::std::os::raw::c_int;
}

extern "C" {
    pub fn MPI_Comm_rank(comm: MPI_Comm, rank: *mut ::std::os::raw::c_int)
        -> ::std::os::raw::c_int;
}

extern "C" {
    pub fn MPI_Comm_size(comm: MPI_Comm, size: *mut ::std::os::raw::c_int)
        -> ::std::os::raw::c_int;
}
