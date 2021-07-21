#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4096M   # memory per CPU core
#SBATCH -J "sum_pair_triples_dt0p05"
#SBATCH --array=10,15  #Corresponds to the timerange


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load python/3.7
module load cuda/10.0
module load cudnn/7.5
module load python-tensorflow-gpu/2.0

python3 -u dataprocessing.py
