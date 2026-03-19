#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=100Samples_ImageNetModel_CurvelanesData
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=100Samples_ImageNetModel_CurvelanesData.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "Experiment: 100Samples_ImageNetModel_CurvelanesData"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

echo "Starting ImageNet Model -> Curvelanes Data Test..."

python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --source_test_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --src_samples 100 \
    --tgt_samples 100 \
    --ratio_src_samples 100 \
    --ratio_tgt_samples 0 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 4096 \
    --modelStr "ImageNet" \
    --file_name "100Samples_ImageNetModel_CurvelanesData.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
