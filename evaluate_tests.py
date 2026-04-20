import os
import sys
import json
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import importlib
import cv2
import numpy as np
import torch

# 1. Environment Setup
CLRERNET_ROOT = "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection"
sys.path.insert(0, CLRERNET_ROOT)
os.chdir(CLRERNET_ROOT)

print("--- Force-registering all CLRerNet custom modules ---")
libs_path = Path("libs")
if libs_path.exists():
    for py_file in libs_path.rglob("*.py"):
        if py_file.name == "setup.py": continue
        module_name = str(py_file.with_suffix('')).replace(os.sep, '.')
        try:
            importlib.import_module(module_name)
        except: pass

from mmengine.config import Config
from mmengine.registry import TRANSFORMS
from mmdet.apis import init_detector, inference_detector

# ====================================================================
# THE FAILSAFE: FORCED RESIZING & KEY INJECTION
# ====================================================================

@TRANSFORMS.register_module(name='InjectCULaneKeys', force=True)
class InjectCULaneKeys:
    def __call__(self, results):
        path = results.get('img_path', results.get('filename', 'dummy.jpg'))
        results.update({'img_path': path, 'filename': path, 'sub_img_name': os.path.basename(path)})
        
        if 'img' not in results:
            img = cv2.imread(path) if os.path.exists(path) else np.zeros((590, 1640, 3), dtype=np.uint8)
        else:
            img = results['img']

        target_w, target_h = 800, 320
        img = cv2.resize(img, (target_w, target_h))
        
        results['img'] = img
        results['ori_shape'] = (target_h, target_w)
        results['img_shape'] = (target_h, target_w)
        results['pad_shape'] = (target_h, target_w)
        results['scale_factor'] = (1.0, 1.0)
        return results

@TRANSFORMS.register_module(name='albumentation', force=True)
class FallbackAlbumentation:
    def __init__(self, **kwargs): pass
    def __call__(self, results): return results

# ====================================================================
# METRIC UTILITIES
# ====================================================================

def load_culane_gt(img_path):
    """Loads ground truth coordinates from .lines.txt file."""
    gt_path = img_path.replace('.jpg', '.lines.txt')
    lanes = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                coords = [float(x) for x in line.strip().split()]
                if len(coords) > 2:
                    # Convert list of x,y to list of (x,y) pairs
                    lanes.append(np.array(coords).reshape(-1, 2))
    return lanes

def calculate_metrics(pred_lanes, gt_lanes, dist_threshold=30):
    """
    Simplified matching logic. 
    A lane is a True Positive (TP) if its average distance to a GT lane is small.
    """
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in pred_lanes:
        found_match = False
        # Pred in CLRerNet is usually a list of points or a specific object
        # We assume pred is an array of points for matching
        p_coords = pred if isinstance(pred, np.ndarray) else np.array(pred)
        
        for i, gt in enumerate(gt_lanes):
            if i in matched_gt: continue
            
            # Simple average distance matching (can be replaced by IoU for more precision)
            # We interpolate to match lengths if necessary
            dist = np.mean(np.linalg.norm(p_coords[:10, :2] - gt[:10, :2], axis=1)) 
            if dist < dist_threshold:
                tp += 1
                matched_gt.add(i)
                found_match = True
                break
        
        if not found_match:
            fp += 1
            
    fn = len(gt_lanes) - len(matched_gt)
    return tp, fp, fn

# ====================================================================

ROOT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
LOGS_DIR = os.path.join(ROOT_DIR, "logs/Exodo")
CONFIG_PATH = os.path.join(CLRERNET_ROOT, "configs/clrernet/culane/clrernet_culane_dla34_ema.py")
CHECKPOINT_PATH = os.path.join(CLRERNET_ROOT, "clrernet_culane_dla34_ema.pth")
OUTPUT_CSV = os.path.join(ROOT_DIR, "logs/Exodo/master_evaluation_results.csv")
DEVICE = 'cuda:0'

def get_lane_stats(model, image_paths):
    total_tp, total_fp, total_fn = 0, 0, 0
    correct_count_images = 0
    detected_counts = []

    for img_path in image_paths:
        if not os.path.exists(img_path): continue
        try:
            result = inference_detector(model, img_path)
            
            # Extract Predictions
            if hasattr(result, 'pred_instances'):
                preds = result.pred_instances.lanes if hasattr(result.pred_instances, 'lanes') else result.pred_instances
            else:
                preds = result[0] if isinstance(result, list) else result
            
            # Extract GT
            gts = load_culane_gt(img_path)
            
            # Match
            tp, fp, fn = calculate_metrics(preds, gts)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if len(preds) == len(gts):
                correct_count_images += 1
            
            detected_counts.append(len(preds))
            
        except Exception as e:
            continue

    # Calculate final aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = correct_count_images / len(image_paths) if image_paths else 0

    return {
        "avg_lanes": np.mean(detected_counts) if detected_counts else 0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "count_accuracy": accuracy,
        "processed_count": len(detected_counts)
    }

def main():
    print(f"\n--- Loading Configuration ---")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.test_pipeline = [dict(type='InjectCULaneKeys'), dict(type='PackDetInputs')]
        
    print(f"\n--- Loading CLRerNet onto {DEVICE} ---")
    model = init_detector(cfg, CHECKPOINT_PATH, device=DEVICE)

    if hasattr(model, 'data_preprocessor'):
        model.data_preprocessor.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
        model.data_preprocessor.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)

    all_rows = []
    json_files = glob.glob(os.path.join(LOGS_DIR, "*.json"))

    for json_path in json_files:
        filename = Path(json_path).stem
        with open(json_path, 'r') as f:
            try: data = json.load(f)
            except: continue

        for exp in data.get("experiments", []):
            args = exp.get("arguments", {})
            
            target_keys = [("Sanity Check", "sanity_check"), ("Data Shift Test Data", "test_run")]
            
            for key, run_type in target_keys:
                if key not in exp["data"]: continue
                
                if key == "Sanity Check":
                    runs = [{"Image Paths": exp["data"][key]["Image Paths"], "Run": -1, "Seed": "N/A", "Results": exp["data"][key]}]
                else:
                    runs = exp["data"][key].get("Individual Test Data", [])

                for run in tqdm(runs, desc=f"Eval {filename} ({run_type})", leave=False):
                    m = get_lane_stats(model, run.get("Image Paths", []))
                    res = run.get("Results", {})
                    
                    row = {**args, "run_type": run_type, "run_idx": run.get("Run"), "seed": run.get("Seed"),
                           "avg_lanes_detected": m["avg_lanes"], 
                           "f1_score": m["f1_score"],
                           "precision": m["precision"],
                           "recall": m["recall"],
                           "accuracy_count_match": m["count_accuracy"],
                           "images_in_sample": m["processed_count"],
                           "mmd_stat": res.get("MMD", {}).get("Stat"), 
                           "mmd_p_value": res.get("MMD", {}).get("P-Value"),
                           "mmd_shift_detected": res.get("MMD", {}).get("Shift Detected")}
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! Comprehensive metrics saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
