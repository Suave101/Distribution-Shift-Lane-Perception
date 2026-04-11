#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=10000sCurveLanes32d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/Exodo/CurveLanes/32d/10000sCurveLanes32d.log"
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: Florida Tech HPC"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate ml_project
cd /home1/adoyle2025/Distribution-Shift-Lane-Perception

# Curvelanes Train to Curvelanes Train
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"

# Curvelanes Train to Curvelanes Val
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid/valid.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"

# Curvelanes Train to CULane Test
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"

# CULane Train to CULane Train
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"

# CULane Train to CULane Test
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"

# CULane Train to Curvelanes Valid
python model_experiment.py \
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid/valid.txt \
    --sample_size 10000 \
    --num_runs 100 \
    --block_idx 4 \
    --seed_base 32 \
    --batch_size 64 \
    --latent_dim 32 \
    --permutation_test_iterations 1000 \
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \
    --file_name "10000sCurveLanes32d.json" \
    --modelStr "CurveLanes"
