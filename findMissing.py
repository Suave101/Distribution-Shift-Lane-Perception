import os
import json

# --- Configuration & Paths ---
PROJECT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
BASE_ROOT = os.path.join(PROJECT_DIR, "LocalBash/ModelExperiments/Phase 1")

MODELS = {
    "CULaneModel": "CU_Lane",
    "CurvelanesModel": "CurveLanes",
    "ImageNetModel": "ImageNet",
    "RandomWeightsModel": "Random",
    "AssistTaxiModel": "ASSIST_Taxi"
}
DATASETS = ["CULanes", "Curvelanes", "AssistTaxi"]
SAMPLE_SIZES = [10, 100, 1000]

def is_run_complete(filepath):
    """Returns True if the JSON is fully written and has a valid TPR, False otherwise."""
    if not os.path.exists(filepath):
        return False
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            tpr_value = data["experiments"][0]["data"]["Data Shift Test Data"]["TPR"]
            return True
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        # Catches empty files, half-written JSONs from crashes, or missing keys
        return False

def main():
    missing_commands = []
    
    print("Scanning logs for missing or crashed experiments...")

    for model_folder, model_str in MODELS.items():
        for n in SAMPLE_SIZES:
            # Reconstruct the path to the bash scripts
            exp_output_dir = os.path.join(BASE_ROOT, model_folder, str(n))
            
            for src in DATASETS:
                for tgt in DATASETS:
                    # Match our backwards compatibility naming logic
                    if src == tgt:
                        job_id = f"{n}Samples_{model_str}Model_{src}Data"
                        sh_file_name = f"{src}_{model_str}M{n}.sh"
                    else:
                        job_id = f"{n}Samples_{model_str}Model_{src}2{tgt}Data"
                        sh_file_name = f"{src}2{tgt}_{model_str}M{n}.sh"
                    
                    # File paths
                    json_filepath = os.path.join(LOG_DIR, f"{job_id}.json")
                    sh_filepath = os.path.join(exp_output_dir, sh_file_name)
                    
                    # Check if we need to rerun
                    if not is_run_complete(json_filepath):
                        missing_commands.append(f"sbatch \"{sh_filepath}\"")
                        
    # Write the missing commands to a new executable bash script
    output_script = os.path.join(PROJECT_DIR, "run_missing_experiments.sh")
    
    with open(output_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n".join([f"{cmd}\nsleep 0.1" for cmd in missing_commands]))
        
    os.chmod(output_script, 0o755)
    
    print(f"Found {len(missing_commands)} incomplete or missing jobs.")
    print(f"Generated recovery script: {output_script}")

if __name__ == "__main__":
    main()
