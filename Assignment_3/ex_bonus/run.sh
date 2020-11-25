#!/bin/bash -l
# The -l above is required to get the full environment with modules

# Set the allocation to be charged for this job
# not required if you have set a default allocation
#SBATCH -A edu20.dd2360

# The name of the script is myjob
#SBATCH -J Assignment_3

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 0:10:00

# Number of nodes
#SBATCH --nodes=1
# Number of MPI processes per node
#SBATCH --ntasks-per-node=24

# GPU
#SBATCH -C Haswell
#SBATCH --gres=gpu:K420:1

#SBATCH -o result.csv

echo "size,gird.x,gird.y,tile_size,cpu_time,cublas,gpu_naive,gpu_shared"
./exercise_3.out -s 64 -v
./exercise_3.out -s 128 -v
./exercise_3.out -s 256 -v
./exercise_3.out -s 512 -v
./exercise_3.out -s 1024 -v
./exercise_3.out -s 2048 -v
./exercise_3.out -s 4096 -v
