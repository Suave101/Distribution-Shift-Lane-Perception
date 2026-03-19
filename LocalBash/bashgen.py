import os

# --- Configuration & Paths ---
BASE_ROOT = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/ModelExperiments/Phase 1"
PROJECT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
CONDA_PROFILE = "/home1/adoyle2025/miniconda3/etc/profile.d/conda.sh"

# Model folder name -> modelStr argument
MODELS = {
    "CULaneModel": "CU_Lane",
    "CurvelanesModel": "CurveLanes",
    "ImageNetModel": "ImageNet",
    "RandomWeightsModel": "Random",
    "AssistTaxiModel": "ASSIST_Taxi"
}

# Dataset label -> (Source Dir, Source List Path, Source Test List Path)
DATASETS = {
    "CULanes": (
        "/home1/adoyle2025/Datasets/Datasets/CULane",
        "/home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt"
    ),
    "Curvelanes": (
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes",
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt"
    ),
    "AssistTaxi": (
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi",
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/train.txt",
        "/home1/adoyle2025/Datasets/Datasets/ASSIST-Taxi/test.txt"
    )
}

SAMPLE_SIZES = [10, 100, 1000]

# --- Template ---
SH_TEMPLATE = """#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name={job_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output={job_id}.log
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
echo "Experiment: {job_id}"
echo "----------------------------------------------------"

export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source {conda_sh}

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd {proj_dir}

echo "Starting {model_str} Model -> {data_label} Data Test..."

python model_experiment.py \\
    --source_dir {src_dir} \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes \\
    --source_list_path {src_list} \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --source_test_list_path {src_test_list} \\
    --src_samples {n} \\
    --tgt_samples {n} \\
    --ratio_src_samples {n} \\
    --ratio_tgt_samples 0 \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 4096 \\
    --modelStr "{model_str}" \\
    --file_name "{job_id}.json"

echo "----------------------------------------------------"
echo "Job finished: $(date)"
echo "----------------------------------------------------"
"""

def main():
    sbatch_commands = []

    for model_folder, model_str in MODELS.items():
        for n in SAMPLE_SIZES:
            # Create Folder Structure
            target_dir = os.path.join(BASE_ROOT, model_folder, str(n))
            os.makedirs(target_dir, exist_ok=True)

            for data_label, (src_dir, src_list, src_test_list) in DATASETS.items():
                job_id = f"{n}Samples_{model_str}Model_{data_label}Data"
                file_name = f"{data_label}_{model_str}M{n}.sh"
                file_path = os.path.join(target_dir, file_name)

                content = SH_TEMPLATE.format(
                    job_id=job_id,
                    conda_sh=CONDA_PROFILE,
                    proj_dir=PROJECT_DIR,
                    model_str=model_str,
                    data_label=data_label,
                    src_dir=src_dir,
                    src_list=src_list,
                    src_test_list=src_test_list,
                    n=n
                )

                with open(file_path, "w") as f:
                    f.write(content)
                
                sbatch_commands.append(f"sbatch \"{file_path}\"")

    # Master Run Script
    master_path = os.path.join(PROJECT_DIR, "run_all_experiments.sh")
    with open(master_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n".join(sbatch_commands))
    
    os.chmod(master_path, 0o755)
    print(f"Success! {len(sbatch_commands)} files generated.")
    print(f"Master script: {master_path}")

if __name__ == "__main__":
    main()
