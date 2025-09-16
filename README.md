# dolfin-hpc-codes

## Introduction
FEM codes written in c++ using dolfin-hpc-0.9.5:
https://bitbucket.org/adaptivesimulations/dolfin-hpc/src/master/
dolfin-hpc is developed on the top of dolfin code from legacy FEniCs project.

Some demos can be found in the 'demo' directory of dolfin-hpc-0.9.5.

## Steps to setup and run demo programs:

1) Create a form file, for example, form.ufl.
This file should contain Finite Element weak formulation written in ufl.
Some examples are available in the demo directory of this repo:
https://bitbucket.org/adaptivesimulations/ffc-hpc/src/master/

2) Use ffc to generate c++ header file as:
```
ffc -l dolfin_hpc form.ufl
```
This will generate header file: form.h

3) Create a main.cpp that contain FEM program and include form.h file.

4) Use Makefile to generate executable
```
make
```

5) Run the program:
```
srun -n <np> ./exe <command line arguments if any>
```

## Notes:

1) Internal mesh generators do not work with parallel I/O. It is better to export mesh in .bin format.
   This can be done using 'dolfin_convert' that can be built along with dolfin-hpc.
   ```
   dolfin_convert mesh.xml mesh.bin
   ```

2) If your application requires more than one FE forms, create a separate .ufl file for each.   

## Install dolfin-hpc and dependencies (on LUMI)

1. Install gts and create a module:

- download gts-0.7.6 and install. follow straigthforward installation steps.
- create modulefiles. see for example: https://researchcomputing.princeton.edu/support/knowledge-base/custom-modules
- load gts module in bashrc

2. Parmetis:
Load already available Parmetis module

3. ffc-hpc:
- first load cray-python module so that pip can be used to install this package.
- get the source code from adaptive simulations and install using pip.
- Then set
  ```export PYTHONPATH=$PYTHONPATH:<path-to-ffc-hpc>/lib/python3.11/site-packages```
- set ```PATH``` eniornmental varible pointing to bin

4. ufl-hpc:
- Noth that install ufl-hpc same as ffc-hpc. Do not follow instructions on the ufl-hpc's repository.
- Then set
  ```export PYTHONPATH=$PYTHONPATH:<path-to-ufl-hpc>/lib/python3.11/site-packages```

5. ufc2-hpc:
- Get the source from repository and install as instructed
- Set ```export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:<path-to-ufc2-hpc>```

6. PETSc:
- Use the instruction on lumi docs about easybuild. Lumi support has made things very convenient.
  https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PETSc/PETSc-3.21.5-cpeGNU-24.03-OpenMP/

7) Trilinos:
- Same as PETSc

8) dolfin-hpc:
- ensure gts pkg-confif paths are added to PKG_CONFIG_PATH
- add ```#include <utility>``` in the file 
  ```dolfin-hpc/include/dolfin/fem/UFCCell.h``` to fix some errors
- configure, make, make install dolfin-hpc
- add ```LD_LIBRARY_PATH```

9) dolfin-convert:
- configure, make, make install in ```dolfin-hpc/misc/convert``` dir
