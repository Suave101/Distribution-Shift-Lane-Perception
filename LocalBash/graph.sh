#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=graph.log
#SBATCH --partition=eternity
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
    
# --- Job Execution ---
echo "----------------------------------------------------"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start Time: $(date)"
echo "----------------------------------------------------"


export PYTHONNOUSERSITE=1

echo "Initializing conda for script..."
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate ml_project

echo "Conda environment activated and isolated:"
conda info --env

cd /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash/DRExperiments

echo "Starting Unit Test..."
#!/bin/bash

# Define the root directory to search (defaults to current directory)
SEARCH_DIR=${1:-.}

echo "Checking file parity (SH vs LOG) in: $SEARCH_DIR"
echo "------------------------------------------------"
printf "%-30s | %-5s | %-5s | %-s\n" "Directory" "SH" "LOG" "Status"
echo "------------------------------------------------"

# Find all unique directories containing either .sh or .log files
find "$SEARCH_DIR" -type d | while read -r dir; do
    # Count .sh files (suppressing errors if none exist)
    sh_count=$(find "$dir" -maxdepth 1 -name "*.sh" | wc -l)
    
    # Count .log files
    log_count=$(find "$dir" -maxdepth 1 -name "*.log" | wc -l)

    # Only report if there is at least one script or one log present
    if [ "$sh_count" -gt 0 ] || [ "$log_count" -gt 0 ]; then
        if [ "$sh_count" -eq "$log_count" ]; then
            status="MATCH"
        else
            status="MISMATCH ⚠️"
        fi
        
        # Clean up directory name for display
        display_dir=$(echo "$dir" | cut -c 1-30)
        printf "%-30s | %-5d | %-5d | %-s\n" "$display_dir" "$sh_count" "$log_count" "$status"
    fi
done

echo "------------------------------------------------"
echo "Audit complete."

# python3 visGen2.py \
#   --log-path /home1/adoyle2025/Distribution-Shift-Lane-Perception/logs \
#   --recursive \
#   --output-dir figures_json

echo "Done!"