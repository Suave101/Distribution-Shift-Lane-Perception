#!/bin/bash
# Rerun all Phase 2 experiments — ASSIST-Taxi excluded
#
# Models:      CU_Lane, CurveLanes, ImageNet, Random
# Datasets:    CULane ↔ CULane, CULane ↔ Curvelanes,
#              Curvelanes ↔ CULane, Curvelanes ↔ Curvelanes
# Sample sizes: 10, 100, 1000
# Total jobs:  48  (4 models × 4 data combos × 3 K values)
#
# Results are written to: logs/Phase2_rerun/
# Usage: bash rerun_phase2_no_assist_taxi.sh

set -euo pipefail

BASE_DIR="/home1/adoyle2025/Distribution-Shift-Lane-Perception"
DATASETS="/home1/adoyle2025/Datasets/Datasets"
CONDA_SH="/home1/adoyle2025/miniconda3/etc/profile.d/conda.sh"
RERUN_LOG_DIR="${BASE_DIR}/logs/Phase2_rerun"
SCRIPT_DIR="${RERUN_LOG_DIR}/scripts"

mkdir -p "${RERUN_LOG_DIR}" "${SCRIPT_DIR}"

# --- Model weights strings (must match model_experimentP2.py modelStr choices) ---
MODELS=("CU_Lane" "CurveLanes" "ImageNet" "Random")

# --- Data combos (parallel arrays) ---
COMBO_NAMES=("CULanes" "CULanes2Curvelanes" "Curvelanes2CULanes" "Curvelanes")
SRC_DATASETS=("CULane"     "CULane"     "Curvelanes" "Curvelanes")
TGT_DATASETS=("CULane"     "Curvelanes" "CULane"     "Curvelanes")

# --- Helpers: resolve list paths from dataset name ---
dataset_train_list() {
    case "$1" in
        CULane)     echo "${DATASETS}/CULane/list/train.txt" ;;
        Curvelanes) echo "${DATASETS}/Curvelanes/train/train.txt" ;;
    esac
}

dataset_test_list() {
    # Curvelanes has no separate test split; use train.txt as source_test_list_path
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

            # Generate individual job script
            cat > "${SCRIPT}" <<SCRIPTEOF
#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=${JOBNAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=${RERUN_LOG_DIR}/${JOBNAME}.log
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
    --source_dir ${DATASETS}/${SRC} \\
    --target_dir ${DATASETS}/${TGT} \\
    --source_list_path $(dataset_train_list "${SRC}") \\
    --target_list_path $(dataset_train_list "${TGT}") \\
    --source_test_list_path $(dataset_test_list "${SRC}") \\
    --src_samples ${K} \\
    --tgt_samples ${K} \\
    --ratio_src_samples 0 \\
    --ratio_tgt_samples ${K} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --modelStr "${MODEL}" \\
    --file_location "${RERUN_LOG_DIR}" \\
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
echo "SLURM logs  : ${RERUN_LOG_DIR}/*.log"
echo "JSON results: ${RERUN_LOG_DIR}/*.json"