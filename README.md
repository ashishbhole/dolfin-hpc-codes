# dolfin-hpc-codes
FEM codes written in c++ using dolfin-hpc-0.9.5:
https://bitbucket.org/adaptivesimulations/dolfin-hpc/src/master/
dolfin-hpc is developed on the top of dolfin code from legacy FEniCs project.

Some demos can be found in the 'demo' directory of dolfin-hpc-0.9.5.

Steps to setup a demo:

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

Notes:

1) Internal mesh generators do not work with parallel I/O. It is better to export mesh in .bin format.
   This can be done using 'dolfin_convert' that can be built along with dolfin-hpc.
   ```
   dolfin_convert mesh.xml mesh.bin
   ```

2) If your application requires more than one FE forms, create a separate .ufl file for each.   
