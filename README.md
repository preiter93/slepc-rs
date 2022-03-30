# slepc-rs

## `slepc-rs`: `SLEPc` rust bindings

Currently, this repository is only to demonstrate
how `SLEPc` can be called from Rust.

### Dependencies
- `clang`
- `mpi`
- `petsc` and `slepc`, see Installation section

### Installation

#### `PETSc`
Download [download `PETSc`](https://petsc.org/release/download/). This
crate is tested with v.`3.16.4`.

Install petsc via
```
export PETSC_ARCH=linux-gnu-real-debug
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90
make all check
```
(You can choose a custom arch name, or build without fc, i.e. `--with-fc=0`,
or build together with lapack, i.e. --download-f2cblaslapack)

Then export directory, arch name and library path, i.e.
```
export PETSC_DIR=${HOME}/local/petsc-3.16.4
export PETSC_ARCH=linux-gnu-real-debug
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PETSC_DIR}/${PETSC_ARCH}/lib
```

#### `PETSc` Complex
```
export PETSC_ARCH=linux-gnu-complex-debug
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-scalar-type=complex
make all check
```
(You can choose a custom arch name, or build without fc, i.e. `--with-fc=0`,
or build together with lapack, i.e. --download-f2cblaslapack)

- *Be careful that LD_LIBRARY_PATH has no old links to a build without complex type!*

- Turn on feature `scalar_complex`!

#### `SLEPc`
Download [download `SLEPc`](https://slepc.upv.es/download/). This
crate is tested with v.`3.16.2`.

Install slepc via
```
./configure
make all check
```
Then export directory and library path, i.e.
```
export SLEPC_DIR=${HOME}/local/slepc-3.16.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${SLEPC_DIR}/${PETSC_ARCH}/lib
```

### Other similar crates
- <https://gitlab.com/petsc/petsc-rs>
- <https://github.com/tflovorn/slepc-sys>

### TODO
- Error Handling

### Documentation
- <https://slepc.upv.es/documentation/slepc.pdf>
