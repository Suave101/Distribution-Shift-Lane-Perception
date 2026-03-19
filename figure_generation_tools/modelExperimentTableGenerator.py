import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration & Paths ---
LOG_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/logs" # Note: Ensure your json files are moving here!
OUTPUT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception/ModelExperimentFigures/Phase1"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the matrix axes based on actual modelStr values used in experiments
MODELS = ["ImageNet", "Random", "CU_Lane", "CurveLanes", "ASSIST_Taxi"]
DATASETS = ["CULanes", "Curvelanes", "AssistTaxi"]
SAMPLE_SIZES = [10, 100, 1000]

def extract_tpr_from_json(filepath):
    """Safely extracts TPR from the specific nested JSON structure."""
    if not os.path.exists(filepath):
        return "In Progress"
        
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Structure: "experiments" [{"data" {..., "Data Shift Test Data": {... "TPR": 3.0,...}}}]
            tpr_value = data["experiments"][0]["data"]["Data Shift Test Data"]["TPR"]
            return float(tpr_value)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        # Handle partially written, empty, or malformed JSONs as "In Progress"
        return "In Progress"

def main():
    sns.set_theme(style="whitegrid")

    # Generate the Y-axis row labels (every permutation)
    row_labels = []
    for src in DATASETS:
        for tgt in DATASETS:
            row_labels.append(f"{src} \u2192 {tgt}") # Uses arrow symbol for "Source -> Target"

    for n in SAMPLE_SIZES:
        print(f"Processing Phase 1 - {n} Samples...")
        
        # 1. Initialize empty results grid (Permutations as Rows, Models as Columns)
        results_grid = {model: {row: "In Progress" for row in row_labels} for model in MODELS}
        
        # 2. Parse Logs and populate grid
        for model in MODELS:
            for src in DATASETS:
                for tgt in DATASETS:
                    # Match the backwards compatibility logic from our generator script
                    if src == tgt:
                        job_id = f"{n}Samples_{model}Model_{src}Data"
                    else:
                        job_id = f"{n}Samples_{model}Model_{src}2{tgt}Data"
                    
                    filename = f"{job_id}.json"
                    filepath = os.path.join(LOG_DIR, filename)
                    
                    row_key = f"{src} \u2192 {tgt}"
                    tpr = extract_tpr_from_json(filepath)
                    results_grid[model][row_key] = tpr
        
        # 3. Convert to Pandas DataFrame (Rows=Permutations, Columns=Models)
        df = pd.DataFrame(results_grid)
        df = df.reindex(index=row_labels, columns=MODELS)
        
        # --- 4. Generate LaTeX Table ---
        latex_df = df.copy()
        for col in latex_df.columns:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

        try:
            latex_str = latex_df.style.to_latex(
                caption=f"Phase 1 TPR (\\%) Results across Distribution Shifts - {n} Samples", 
                label=f"tab:tpr_{n}",
                hrules=True
            )
        except AttributeError:
            latex_str = latex_df.to_latex(
                caption=f"Phase 1 TPR (\\%) Results across Distribution Shifts - {n} Samples", 
                label=f"tab:tpr_{n}"
            )
            
        tex_filepath = os.path.join(OUTPUT_DIR, f"Table_Phase1_{n}Samples.tex")
        with open(tex_filepath, "w") as f:
            f.write(latex_str)
        print(f"  -> Saved LaTeX Table: {tex_filepath}")
        
        # --- 5. Generate Figure (Heatmap) ---
        df_numeric = df.replace("In Progress", np.nan).apply(pd.to_numeric)
        
        # Increased height to 9 to accommodate all 9 permutation rows cleanly
        plt.figure(figsize=(12, 9))
        
        ax = sns.heatmap(
            df_numeric, 
            annot=True,          
            fmt=".2f",          
            cmap="viridis",      
            vmin=0,             
            vmax=100,            
            linewidths=.5,       
            cbar_kws={'label': 'True Positive Rate (TPR %)'} 
        )
        
        plt.title(f"Phase 1 Distribution Shift: TPR (%) - {n} Samples\n(Blank cells indicate 'In Progress')", fontsize=14)
        plt.xlabel("Model Pretraining Weights", fontsize=12)
        plt.ylabel("Evaluation (Source \u2192 Target)", fontsize=12)
        
        # Rotate Y-axis labels so they are readable
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        fig_filepath = os.path.join(OUTPUT_DIR, f"Figure_Phase1_{n}Samples.png")
        plt.savefig(fig_filepath, dpi=300) 
        plt.close()
        print(f"  -> Saved Heatmap Figure: {fig_filepath}")

if __name__ == "__main__":
    main()
