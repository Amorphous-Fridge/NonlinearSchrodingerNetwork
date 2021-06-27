#!/bin/bash

#SBATCH --time=01:00:00   # walltime
# --exclude=m8-18-6,m8-11-2,m7-1-1,m7-4-3,m7-8-11,m8g-1-1,m8g-3-11,m8g-2-9,m7-10-3,m7-11-5,m7-5-5,m7-10-5,m8-1-1,
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --exclusive=user
#SBATCH --mem=1gb
#SBATCH -J "3layers distributed"
#SBATCH --mail-user=mitchellccutler@gmail.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH --parsable

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export WIDTH=$SLURM_ARRAY_TASK_ID
module load python/3.8

python3 -u gridsearch_3layers.py
