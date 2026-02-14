#!/usr/bin/env python3
"""
Visualize Mixed Shift Experiment Results
- Title: Distribution Shift Study
- Subtitle: {Parent Folder Name} | {Dims} Dimensions | Sample Size: {K}
- Sorting: Clean Data (Left) -> 100% Shift (Right, Exp 1)
- Architecture string footer with bottom padding.
"""

import re
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global Configuration
plt.rcParams['figure.figsize'] = (18, 10)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

ARCH_STRINGS = {
    "d128rel": "4096 -> 1024 -> 128", "d64rel": "4096 -> 1024 -> 64", "d32rel": "4096 -> 1024 -> 32",
    "d128gdd": "4096 -> 512 -> 128", "d64gdd": "4096 -> 512 -> 64", "d32gdd": "4096 -> 512 -> 32",
    "d64ids": "4096 -> 1024 -> 256 -> 64", "d128ids": "4096 -> 1024 -> 256 -> 128",
    "orig": "4096 -> 1024 -> 256 -> 128", "d32": "4096 -> 1024 -> 256 -> 32",
}

def camel_to_title(text):
    """Converts CamelCase or underscore_strings to Title Case (e.g. 'GraduallyDecrease' -> 'Gradually Decrease')"""
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert space before capital letters if not already preceded by space
    return re.sub(r'(?<!^)(?<!\s)(?=[A-Z])', ' ', text).strip()

def parse_bash_script(bash_path):
    if not bash_path.exists(): return {}
    try:
        with open(bash_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception: return {}
    
    config = {}
    dconfig_match = re.search(r'--dConfig\s+"([^"]+)"', content)
    dconf = dconfig_match.group(1) if dconfig_match else "orig"
    
    config['arch_key'] = dconf
    
    # Extract Dimension Count
    dim_match = re.search(r'd(\d+)', dconf)
    if dim_match:
        config['dims'] = dim_match.group(1)
    else:
        config['dims'] = "32" if "d32" in dconf else "128"
    
    return config

def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception: return None
    
    metrics = {}
    # Sample Size K usually matches total target pool or initial setup
    k_match = re.search(r'K(\d+)', log_path.stem)
    metrics['sample_size'] = k_match.group(1) if k_match else "Unknown"

    tau = re.search(r'\[RESULT\] τ\([\d.]+\) = ([\d.]+)', content)
    mmd = re.search(r'Average MMD: ([\d.]+) ± ([\d.]+)', content)
    tpr = re.search(r'TPR \(true positive rate\) over \d+ runs: ([\d.]+)%', content)
    shift = re.search(r'\[STEP 3\] Data Shift Test:.*?\((\d+)\).*?Curvelanes.*?\((\d+)\).*?CULane', content)
    
    if tau: metrics['tau'] = float(tau.group(1))
    if mmd: metrics['avg_mmd'], metrics['std_mmd'] = float(mmd.group(1)), float(mmd.group(2))
    if tpr: metrics['tpr'] = float(tpr.group(1))
    if shift: 
        metrics['tgt_samples'], metrics['src_samples'] = int(shift.group(1)), int(shift.group(2))
    
    return metrics if 'avg_mmd' in metrics else None

def load_folder_data(logs_dir):
    experiments = []
    path = Path(logs_dir)
    log_files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == '.log'])
    
    # Capture Parent Folder Name for the Title (e.g. "GraduallyDecreaseDimensions")
    # If logs are in .../GraduallyDecreaseDimensions/thousand64d/, parent is 'GraduallyDecreaseDimensions'
    parent_folder_name = path.parent.name
    
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        if metrics:
            bash_file = path / f"{log_file.stem}.sh"
            metrics['bash'] = parse_bash_script(bash_file)
            metrics['parent_folder'] = parent_folder_name # Store for plotting
            experiments.append(metrics)
    
    # SORTING: Clean (High Src) -> Shifted (Low Src)
    experiments.sort(key=lambda x: x.get('src_samples', 0), reverse=False)
    for i, exp in enumerate(experiments):
        exp['id'] = i + 1
    
    experiments.sort(key=lambda x: x.get('src_samples', 0), reverse=True)
    return experiments

def plot_comprehensive_analysis(experiments, output_dir, folder_name):
    if not experiments: return
    n = len(experiments)
    x = np.arange(n)
    
    # Extract Plotting Data
    tprs = [e.get('tpr', 0) for e in experiments]
    mmds = [e.get('avg_mmd', 0) for e in experiments]
    stds = [e.get('std_mmd', 0) for e in experiments]
    taus = [e.get('tau', 0) for e in experiments]
    srcs = [e.get('src_samples', 0) for e in experiments]
    tgts = [e.get('tgt_samples', 0) for e in experiments]
    ids = [e.get('id', 0) for e in experiments]
    ratios = [tgts[i] / max(1, srcs[i]) for i in range(n)]
    
    # Metadata for labels
    config = experiments[0].get('bash', {})
    
    # Title Logic: Use Parent Folder Name
    raw_title = experiments[0].get('parent_folder', "Unknown")
    pretty_arch = camel_to_title(raw_title)
    
    dims = config.get('dims', "???")
    sample_size = experiments[0].get('sample_size', "Unknown")
    arch_str = ARCH_STRINGS.get(config.get('arch_key', 'orig'), "Architecture Not Found")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8.8))

    # Panel 1: TPR
    colors = ['#06A77D' if t == 100 else '#D62828' for t in tprs]
    ax1.bar(x, tprs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax1.set_title('True Positive Rate (%)', fontweight='bold', fontsize=14)
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Exp {i}" for i in ids], rotation=45)

    # Panel 2: MMD vs Threshold
    ax2.errorbar(x, mmds, yerr=stds, fmt='o-', color='#F18F01', linewidth=2, markersize=8, label='Avg MMD', capsize=4)
    ax2.step(x, taus, where='mid', color='#2E86AB', linestyle='--', linewidth=2, label='Threshold (τ)')
    ax2.set_title('MMD Metric vs τ', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r:.2f}R" if r < 100 else "MaxR" for r in ratios], rotation=45)

    # Panel 3: Sample Sizes
    width = 0.35
    ax3.bar(x - width/2, srcs, width, label='Source (Clean)', color='#06A77D', alpha=0.8)
    ax3.bar(x + width/2, tgts, width, label='Target (Shift)', color='#F18F01', alpha=0.8)
    ax3.set_title('Sample Sizes', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Exp {i}" for i in ids], rotation=45)

    # Titles and Subtitles
    plt.suptitle("Distribution Shift Study", fontsize=26, fontweight='bold', y=0.98)
    
    # Subtitle now uses the folder-based title
    subtitle_str = f"{pretty_arch} | {dims} Dimensions | Sample Size: {sample_size}"
    plt.figtext(0.5, 0.92, subtitle_str, ha='center', fontsize=18, fontstyle='italic', color='#333333')

    # Architecture Footer
    fig.text(0.5, 0.03, f"Network Architecture: {arch_str}", ha='center', fontsize=15, fontweight='bold', 
             bbox=dict(facecolor='#FBFCFC', alpha=0.9, edgecolor='#AEB6BF', boxstyle='round,pad=1.2'))

    plt.tight_layout(rect=[0, 0.15, 1, 0.90])
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f"{folder_name}_summary.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='figures')
    parser.add_argument('--all-dirs', action='store_true')
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.base_path):
        if any(f.endswith('.log') for f in files):
            log_dir = Path(root)
            data = load_folder_data(log_dir)
            if data:
                folder_id = str(log_dir.relative_to(args.base_path)).replace('/', '_').replace('\\', '_')
                if not folder_id or folder_id == ".": folder_id = log_dir.name
                plot_comprehensive_analysis(data, args.output_dir, folder_id)
                print(f"✓ Generated: {folder_id}")

if __name__ == "__main__":
    main()