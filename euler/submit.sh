#!/bin/bash -l
#SBATCH -A naiss2025-5-152
#SBATCH -J cylinder
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -e error_file.e
#SBATCH -o output_file.o
#SBATCH -p main
srun ./demo

