#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=Build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception/localBash/buildLog.log
#SBATCH --partition=long
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=START,END,FAIL

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

# =========================================================
# Step 1: Load Environment Modules
# =========================================================
start_step "Loading Apptainer Module"
module load apptainer/1.4.1-gcc-13.3.0-nmn2ykt
end_step

# =========================================================
# Step 2: Navigate to Directory
# =========================================================
start_step "Navigating to Directory"
cd /home1/adoyle2025/suave101/Distribution-Shift-Lane-Perception/
end_step

# =========================================================
# Step 3: Build Container
# =========================================================
start_step "Building Apptainer Container (Nuitka Compilation)"
apptainer build --force --fakeroot shift-detector.sif shift-detector.def
end_step

# =========================================================
# Step 4: Run Container
# =========================================================
start_step "Running Apptainer Executable"
apptainer run shift-detector.sif
end_step

# =========================================================
# Final Timing Summary
# =========================================================
JOB_END=$(date +%s)
TOTAL_ELAPSED=$((JOB_END - JOB_START))
TOTAL_MINS=$((TOTAL_ELAPSED / 60))
TOTAL_SECS=$((TOTAL_ELAPSED % 60))

echo "=========================================================="
echo "ALL STEPS SUCCESSFUL!"
echo "TOTAL JOB RUNTIME: ${TOTAL_MINS}m ${TOTAL_SECS}s (${TOTAL_ELAPSED} seconds)"
echo "=========================================================="

