#!/bin/bash
# ==============================================================================
# AI.Panther HPC Job Submission Script - Florida Institute of Technology
# ==============================================================================
#SBATCH --job-name=curvelanes_shift
#SBATCH --partition=long               # FIT partition (e.g., short, batch)
#SBATCH --nodes=1                      # Run on a single compute node
#SBATCH --ntasks=1                     # Single process task
#SBATCH --cpus-per-task=16             # Allocate 8 CPU threads
#SBATCH --mem=32GB                     # Explicit RAM request (Required on AI.Panther)
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --output=logs/shift_%j.log     # Standard output log (%j = Slurm Job ID)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adoyle2025@my.fit.edu  # FIT TRACKS email notification

# Exit immediately if a command fails
set -e

# =========================================================
# Timing Helper Functions
# =========================================================
start_step() {
    STEP_NAME="$1"
    STEP_START=$(date +%s)
    echo "=========================================================="
    echo "[$(date +'%H:%M:%S')] STARTING STEP: $STEP_NAME"
    echo "=========================================================="
}

end_step() {
    STEP_END=$(date +%s)
    ELAPSED=$((STEP_END - STEP_START))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    echo "----------------------------------------------------------"
    echo "COMPLETED: $STEP_NAME in ${MINS}m ${SECS}s (${ELAPSED} seconds)"
    echo "=========================================================="
    echo ""
}

# Track entire job execution time
JOB_START=$(date +%s)

# Record start time
echo "=========================================================="
echo "Starting job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=========================================================="

# Create log directory if it doesn't exist
mkdir -p logs

# 1. Load Apptainer module
module load apptainer/1.4.1-gcc-13.3.0-nmn2ykt

# 2. Navigate to your project directory in /home1/
cd /home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception/

# 3. Execute experiment inside the container

export PYTHONPATH=/home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception:$PYTHONPATH

mkdir -p /home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception/experiment_output

start_step "Running Apptainer Executable Experiment"
apptainer exec \
    --nv \
    --bind /home1/adoyle2025:/home1/adoyle2025 \
    shift-detector.sif \
    /app/experiment.dist/shift_detector \
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid \
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid/valid.txt \
    --num_runs 100 \
    --sample_size 1000 \
    --batch_size 1000 \
    --permutation_test_iterations 0 \
    --file_location /home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception/experiment_output \
    imagenet_weights
end_step

echo "=========================================================="
echo "Job finished successfully at $(date)"
echo "=========================================================="
