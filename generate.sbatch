#!/bin/bash

# email on start, end, and abortion
#SBATCH --job-name=dataset_generate

#SBATCH --output=out.out
#SBATCH --error=error.out
#SBATCH --partition=caslake
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --account=pi-andrewferguson
#SBATCH --cpus-per-task=4  # SET NUM THREADS (cores)
#SBATCH --mem=8G
# THIS EXAMPLE USES 1 GPU NODE - 1 MPI TASK - 4 THREADS PER TASK

# SET NUMBER OF MPI TASKS 
# SET NUMBER OF MD STEPS

#LOAD GROMACS MODULE 


source activate /project2/andrewferguson/berlaga/conda_env
module load gromacs
#module load cuda/10.0
#module load openmpi/2.0.2
#source /project2/andrewferguson/Kirill/plumed_mods/plumed-2.5.2/sourceme.sh
#source /project2/andrewferguson/Kirill/gromacs_2019.2/bin/GMXRC
python generate.py
