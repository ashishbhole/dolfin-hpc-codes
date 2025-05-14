#!/bin/bash -l
#SBATCH -A naiss2025-5-152
#SBATCH -J cylinder
#SBATCH -t 06:00:00
#SBATCH --ntasks-per-node=8
#SBATCH -n 8
#SBATCH -e error_file.e
#SBATCH -o output_file.o
#SBATCH -p shared
srun -n 8 --ntasks-per-node 8 ./demo
