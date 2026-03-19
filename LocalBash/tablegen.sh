#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=tablegen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=table.log
#SBATCH --partition=eternity
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"


export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception/figure_generation_tools

echo "Starting Table Gen..."

python3 modelExperimentTableGenerator.py

echo "Done!"