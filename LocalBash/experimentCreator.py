import os

# --- Configuration ---
models = ["ImageNet", "Random", "CU_Lane", "CurveLanes", "ASSIST_Taxi", "DISTILL"]
sample_sizes = [10000, 1000, 100, 10]
dimensions = [256, 128, 32]

base_bash_dir = "LocalBash/Exodo"
run_script_path = "run.sh"

# --- Bash Template ---
def get_bash_template(model, size, dim):
    return f"""#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name={size}s{model}{dim}d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="/home1/adoyle2025/Distribution-Shift-Lane-Perception/{base_bash_dir}/{model}/{dim}d/{size}s{model}{dim}d.log"
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
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"

# Curvelanes Train to Curvelanes Val
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid/valid.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"

# Curvelanes Train to CULane Test
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/train \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"

# CULane Train to CULane Train
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"

# CULane Train to CULane Test
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/test.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"

# CULane Train to Curvelanes Valid
python model_experiment.py \\
    --source_dir /home1/adoyle2025/Datasets/Datasets/CULane \\
    --target_dir /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid \\
    --source_list_path /home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt \\
    --target_list_path /home1/adoyle2025/Datasets/Datasets/Curvelanes/valid/valid.txt \\
    --sample_size {size} \\
    --num_runs 100 \\
    --block_idx 4 \\
    --seed_base 32 \\
    --batch_size 64 \\
    --latent_dim {dim} \\
    --permutation_test_iterations 1000 \\
    --file_location "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/" \\
    --file_name "{size}s{model}{dim}d.json" \\
    --modelStr "{model}"
"""

# --- Generation Logic ---
def main():
    run_script_lines = ["#!/bin/bash\n", "echo 'Submitting all SLURM jobs...'\n"]

    for model in models:
        run_script_lines.append(f"\n# =========================================")
        run_script_lines.append(f"# {model} Model Jobs")
        run_script_lines.append(f"# =========================================")
        
        for dim in dimensions:
            dir_path = os.path.join(base_bash_dir, model, f"{dim}d")
            os.makedirs(dir_path, exist_ok=True)
            
            for size in sample_sizes:
                # Updated to include dimension in filename
                filename = f"{size}s{model}{dim}d.sh"
                filepath = os.path.join(dir_path, filename)
                
                script_content = get_bash_template(model, size, dim)
                with open(filepath, "w") as f:
                    f.write(script_content)
                
                bash_path = f"{base_bash_dir}/{model}/{dim}d/{filename}"
                run_script_lines.append(f"sbatch {bash_path}")

    with open(run_script_path, "w") as f:
        f.write("\n".join(run_script_lines) + "\n")
        
    os.chmod(run_script_path, 0o755)

    print(f"✅ Generated 72 scripts inside '{base_bash_dir}/'")
    print(f"✅ Generated master submission script '{run_script_path}'")
    print(f"-> You can now run: ./run.sh")

if __name__ == "__main__":
    main()
