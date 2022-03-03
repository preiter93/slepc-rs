static char help[] = "Simple Hello World example program in SLEPc\n";

#include "slepcsys.h"

int main( int argc, char **argv )
{
  int ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Hello world\n");CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
