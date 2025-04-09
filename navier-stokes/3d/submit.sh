#!/bin/bash -l
#SBATCH -A naiss2024-5-25
#SBATCH -J cylinder
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -n 1
#SBATCH -e error_file.e
#SBATCH -o output_file.o
#SBATCH -p main
srun -n 8 --ntasks-per-node 8 ./demo
