import os
import numpy as np
from math import log
os.system("make clean")
os.system("ffc -l dolfin_hpc Poisson.ufl")
os.system("ffc -l dolfin_hpc Projection.ufl")
os.system("make")
os.system("rm -rf error.dat")
# path to executable
exe = './demo'
# check that executable is present
if os.path.isfile(exe)==False:
    print("Could not find ", exe)
    exit()
N = [0,1,2,3,4]
for rl in N:
   os.system("srun -n 1 "+exe+" "+str(rl)+" "+" > run.log")
   os.system("tail -1 run.log >> error.dat")

# read error from file
data = np.loadtxt('error.dat')
print("---------------------------------------")
print("#h, err_l1, err_l2, err_linf, L2_rate")
for j in range(5):
   if j==0:
      fmt='{1:14.6e} {1:14.6e} {2:14.6e} {3:14.6e}'
      print(fmt.format(data[j][0], data[j][1], data[j][2], data[j][3]))
   else:
      rate_L2 = log(data[j-1][2]/data[j][2])/log(2)
      fmt='{1:14.6e} {1:14.6e} {2:14.6e} {3:14.6e} {4:10.3f}'
      print(fmt.format(data[j][0], data[j][1], data[j][2], data[j][3], rate_L2))
print("---------------------------------------")
