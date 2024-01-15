This demo:
1) Defines a space-time dependent function
2) Projects in onto chosen FE space in Projection.ufl
3) Evaluates the FE function at a prescirbed point in a computational domain

When running with multiple processors, MPI_Reduce operation is needed.

How to run:
```
make clean
```
Use ffc compiler on a form file
```
ffc -l dolfin_hpc Projection.ufl
```
This will generate Projection.h file and it is included in main.cpp. 
Then build the demo program to generate the executable 'demo'.
```
make
```
Then run on HPC after allocating resources
```
srun ./demo
```
