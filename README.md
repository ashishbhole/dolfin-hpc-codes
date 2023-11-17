# dolfin-hpc-codes
FEM codes written in c++ using dolfin-hpc-0.9.5:
https://bitbucket.org/adaptivesimulations/dolfin-hpc/src/master/

dolfin-hpc is developed on the top of dolfin code from legacy FEniCs project.

Steps:

1) Create a form file in ufl as form.ufl. Some examples are available in the demo directory of this repo:
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

Notes:

1) Internal mesh generators do not work with parallel I/O. It is better to export mesh in .bin format.
   This can be done using dolfin_convert.
   ```
   dolfin_convert mesh.xml mesh.bin
   ```
