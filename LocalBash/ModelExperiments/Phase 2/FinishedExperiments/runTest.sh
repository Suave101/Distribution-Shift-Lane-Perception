#!/bin/bash
# Run Phase 2 "Test-target" experiments — fills in the four missing Test rows of
# the Phase 2 heatmap (CULane/Curvelanes source vs. CULane/Curvelanes TEST target).
#
# For each combination of:
#   Models      (4): CU_Lane, CurveLanes, ImageNet, Random
#   Sample sizes(3): 10, 100, 1000
# the following 4 data configurations are run:
#   CULane     → CULane      Test
#   CULane     → Curvelanes  Test
#   Curvelanes → CULane      Test
#   Curvelanes → Curvelanes  Test
#
# Total jobs: 4 models × 4 combos × 3 K values = 48
#
# Results are written to: logs/Phase2_test/
# Usage: bash LocalBash/run_phase2_test.sh

set -euo pipefail

BASE_DIR="/home1/adoyle2025/Distribution-Shift-Lane-Perception"
DATASETS="/home1/adoyle2025/Datasets/Datasets"
CONDA_SH="/home1/adoyle2025/miniconda3/etc/profile.d/conda.sh"
TEST_LOG_DIR="${BASE_DIR}/logs/Phase2_test"
SCRIPT_DIR="${TEST_LOG_DIR}/scripts"

mkdir -p "${TEST_LOG_DIR}" "${SCRIPT_DIR}"

# --- Model weights strings (must match model_experimentP2.py modelStr choices) ---
MODELS=("CU_Lane" "CurveLanes" "ImageNet" "Random")

# --- Data combos (parallel arrays) ---
# These mirror the four Train combos but with test-split target list paths.
# Curvelanes has no separate test split, so it reuses train/train.txt as the
# target list (consistent with bashgen.py DATA_MAP).
COMBO_NAMES=("CULanesTest" "CULanes2CurvelanesTest" "Curvelanes2CULanesTest" "CurvelanesTest")
SRC_DATASETS=("CULane"     "CULane"     "Curvelanes" "Curvelanes")
TGT_DATASETS=("CULane"     "Curvelanes" "CULane"     "Curvelanes")

# --- Helpers: resolve paths from dataset name ---
dataset_dir() {
    case "$1" in
        CULane)     echo "${DATASETS}/CULane" ;;
        Curvelanes) echo "${DATASETS}/Curvelanes" ;;
    esac
}

dataset_train_list() {
    case "$1" in
        CULane)     echo "${DATASETS}/CULane/list/train.txt" ;;
        Curvelanes) echo "${DATASETS}/Curvelanes/train/train.txt" ;;
    esac
}

# Source-side sanity-check list (used for calibration only).
# Curvelanes does not have a separate test split; falls back to train.txt.
dataset_src_test_list() {
    case "$1" in
        CULane)     echo "${DATASETS}/CULane/list/test.txt" ;;
        Curvelanes) echo "${DATASETS}/Curvelanes/train/train.txt" ;;
    esac
}

# Target test-split list — the key difference from the "Train" experiments.
# CULane ships with a dedicated test.txt file.
# Curvelanes does not have a separate test split (its DATA_MAP entry in
# LocalBash/bashgen.py maps both train and test to the same train/train.txt),
# so the train list is reused here as well.
dataset_tgt_test_list() {
    case "$1" in
        CULane)     echo "${DATASETS}/CULane/list/test.txt" ;;
        Curvelanes) echo "${DATASETS}/Curvelanes/train/train.txt" ;;
    esac
}

# --- Submit all jobs ---
SUBMITTED=0

for MODEL in "${MODELS[@]}"; do
    for K in 10 100 1000; do

        [[ $K -eq 1000 ]] && MEM="128G" || MEM="64G"

        for IDX in "${!COMBO_NAMES[@]}"; do
            COMBO="${COMBO_NAMES[$IDX]}"
            SRC="${SRC_DATASETS[$IDX]}"
            TGT="${TGT_DATASETS[$IDX]}"

            JOBNAME="P2${K}Samples_${MODEL}Model_${COMBO}Data"
            SCRIPT="${SCRIPT_DIR}/${JOBNAME}.sh"

            # Generate individual Slurm job script
            cat > "${SCRIPT}" <<SCRIPTEOF
#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=${JOBNAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${TEST_LOG_DIR}/${JOBNAME}.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MEM}

# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: \$SLURM_JOB_ID"
echo "Running on host: \$(hostname)"
echo "Experiment: ${JOBNAME}"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ${CONDA_SH}
conda activate ml_project
cd ${BASE_DIR}

python model_experimentP2.py \\
    --source_dir $(dataset_dir "${SRC}") \\
    --target_dir $(dataset_dir "${TGT}") \\
    --source_list_path $(dataset_train_list "${SRC}") \\
    --target_list_path $(dataset_tgt_test_list "${TGT}") \\
    --source_test_list_path $(dataset_src_test_list "${SRC}") \\
    --src_samples ${K} \\
    --tgt_samples ${K} \\
    --ratio_src_samples 0 \\
    --ratio_tgt_samples ${K} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --modelStr "${MODEL}" \\
    --file_location "${TEST_LOG_DIR}" \\
    --file_name "${JOBNAME}.json"

echo "----------------------------------------------------"
echo "Job finished: \$(date)"
echo "----------------------------------------------------"
SCRIPTEOF

            chmod +x "${SCRIPT}"
            sbatch "${SCRIPT}"
            SUBMITTED=$((SUBMITTED + 1))
            sleep 0.1

        done
    done
done

echo "Submitted ${SUBMITTED} jobs."
echo "Job scripts : ${SCRIPT_DIR}/"
echo "SLURM logs  : ${TEST_LOG_DIR}/*.log"
echo "JSON results: ${TEST_LOG_DIR}/*.json"
