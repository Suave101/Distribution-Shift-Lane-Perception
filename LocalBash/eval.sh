#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=Eval_Exodo_A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/eval_exodo_gpu2_%j.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
# 1. Path to your current project (where the evaluation script is)
export PROJECT_ROOT="/home1/adoyle2025/Distribution-Shift-Lane-Perception"

# 2. Path to the CLRerNet source code (where the 'libs' folder lives)
export CLRERNET_ROOT="/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection"

# 3. Add BOTH to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$CLRERNET_ROOT

cd $PROJECT_ROOT

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate clrernet

# Set workspace config for deterministic CuBLAS (matching your training style)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Execution ---
echo "----------------------------------------------------"
echo "Starting Evaluation on A100 (gpu2)"
echo "Target: All JSONs in /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash"
echo "Host: $(hostname)"
echo "----------------------------------------------------"

# Run the aggregation and lane detection script
python evaluate_tests.py

echo "Evaluation finished at: $(date)"
