import os
import json
import glob
import argparse

def process_directory(base_path):
    # Find all json files in the target directory
    search_pattern = os.path.join(base_path, "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"No JSON files found in {base_path}")
        return

    for json_path in json_files:
        filename = os.path.basename(json_path)
        
        # Skip cache or irrelevant files
        if filename == "image_metrics_cache.json":
            continue
            
        # The directory name for this JSON file (e.g., '10sImageNet128d')
        json_dir_name = os.path.splitext(filename)[0]
        json_dir_path = os.path.join(base_path, json_dir_name)
        
        # Load the JSON
        try:
            with open(json_path, 'r') as f:
                schema = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        experiments = schema.get("experiments", [])
        if not experiments:
            continue
            
        print(f"Processing {filename} -> found {len(experiments)} experiments.")
        
        # Create the top-level directory for the JSON file
        os.makedirs(json_dir_path, exist_ok=True)
        
        # Iterate through each experiment
        for exp_idx, experiment in enumerate(experiments):
            # Get arguments for experiment type
            arg_block = experiment.get("arguments", {})
            source = arg_block["source_dir"].replace("/home1/adoyle2025/Datasets/Datasets/", "").replace("/", "_")
            target = arg_block["target_dir"].replace("/home1/adoyle2025/Datasets/Datasets/", "").replace("/", "_")

            # 1-based indexing for folder names (Experiment_1, Experiment_2, etc.)
            exp_dir_path = os.path.join(json_dir_path, f"{source}_to_{target}")
            os.makedirs(exp_dir_path, exist_ok=True)
            
            # --- Dynamically extract prefixes from arguments ---
            args_block = experiment.get("arguments", {})
            source_dir = args_block.get("source_dir", "")
            target_dir = args_block.get("target_dir", "")
            
            # Build the dynamic list, ignoring empty strings and avoiding duplicates
            dynamic_prefixes = []
            if source_dir:
                dynamic_prefixes.append(source_dir)
            if target_dir and target_dir not in dynamic_prefixes:
                dynamic_prefixes.append(target_dir)
            
            # Use a set to store unique paths for the Concat file
            concat_paths = set() 
            
            data_block = experiment.get("data", {})
            for test_name, test_data in data_block.items():
                individual_tests = test_data.get("Individual Test Data", [])
                
                # Iterate through each run
                for run in individual_tests:
                    run_id = run.get("Run", "Unknown")
                    paths = run.get("Image Paths", [])
                    
                    if not paths:
                        continue
                    
                    clean_paths = []
                    for p in paths:
                        # Clean paths using the dynamic prefixes
                        for prefix in dynamic_prefixes:
                            if p.startswith(prefix):
                                p = p.replace(prefix, "", 1)
                                # Remove any leftover leading slash so it's a true relative path
                                p = p.lstrip('/') 
                                break  # Stop checking after the first matching prefix is found
                        clean_paths.append(p)
                            
                    # 1. Write the individual run list file
                    run_file_path = os.path.join(exp_dir_path, f"Run_{run_id}.txt")
                    with open(run_file_path, 'w') as f:
                        for p in clean_paths:
                            f.write(f"{p}\n")
                            concat_paths.add(p) # Add to the master set for this experiment
            
            # 2. Write the concatenated list file for the experiment
            if concat_paths:
                concat_file_path = os.path.join(exp_dir_path, "full.txt")
                with open(concat_file_path, 'w') as f:
                    for p in sorted(concat_paths):
                        f.write(f"{p}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract image lists into a nested directory structure.")
    parser.add_argument("--dir", default=".", help="Directory containing the JSON files (default: current directory).")
    
    args = parser.parse_args()
    
    process_directory(args.dir)
