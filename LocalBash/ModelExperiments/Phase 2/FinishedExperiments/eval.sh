#!/bin/bash
#SBATCH --job-name=lane_eval
#SBATCH --output=logs/eval64/eval2_%A_%a.out  # %A is the array master ID, %a is the specific task ID
#SBATCH --error=logs/eval64/eval2_%A_%a.err
#SBATCH --array=0-7                   # Exactly 8 jobs: IDs 0 through 7
#SBATCH --nodes=1                     # Each of the 8 jobs gets its own node allocation
#SBATCH --ntasks=1                    
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1                  # <-- FIXED: Using classic GRES syntax to request 1 GPU
#SBATCH --cpus-per-task=4             
#SBATCH --mem=16G                     
#SBATCH --time=04:00:00          

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception/

# Define Arrays
DATASETS=("curvelanes" "culane")
MODELS=("ImageNet" "Random" "CULane" "Curvelanes")

# Calculate which dataset and model this specific Job ID should run
# There are 4 models, so we divide by 4 for the dataset, and mod 4 for the model.
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / 4))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 4))

CURRENT_DATASET=${DATASETS[$DATASET_IDX]}
CURRENT_MODEL=${MODELS[$MODEL_IDX]}

echo "=========================================================="
echo "Starting Slurm Array Task: $SLURM_ARRAY_TASK_ID"
echo "Assigned Dataset: $CURRENT_DATASET"
echo "Assigned Model: $CURRENT_MODEL"
echo "Executing on Node: $SLURMD_NODENAME"
echo "=========================================================="

# Run the python script for this specific dataset and model combination
srun python evaluatePhase2.py --dataset $CURRENT_DATASET --model_name $CURRENT_MODEL

echo "Task $SLURM_ARRAY_TASK_ID complete!"
