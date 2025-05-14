#!/bin/bash -l
#SBATCH -A naiss2025-5-152
#SBATCH -J post_heart
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -p shared
srun /cfs/klemming/projects/snic/heartsolver/joel/dolfin_post -s solution -m meshfile.bin -t vtk -n 9999

