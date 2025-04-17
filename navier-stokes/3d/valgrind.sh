#!/bin/bash -l
#SBATCH -A naiss2025-5-152
#SBATCH -J valgrind
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -n 8
#SBATCH -e valgrind.err
#SBATCH -o valgrind.out
#SBATCH -p main
valgrind4hpc -n 8 --tool=memcheck --outputfile=valgrind_output.txt ./demo
