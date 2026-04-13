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

        # Force 800x320 for DLA-34 stability
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

ROOT_DIR = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
LOGS_DIR = os.path.join(ROOT_DIR, "logs/Exodo")
CONFIG_PATH = os.path.join(CLRERNET_ROOT, "configs/clrernet/culane/clrernet_culane_dla34_ema.py")
CHECKPOINT_PATH = os.path.join(CLRERNET_ROOT, "clrernet_culane_dla34_ema.pth")
OUTPUT_CSV = os.path.join(ROOT_DIR, "logs/Exodo/master_evaluation_results.csv")
DEVICE = 'cuda:0'

def get_lane_stats(model, image_paths):
    counts = []
    for img_path in image_paths:
        if not os.path.exists(img_path): continue
        try:
            result = inference_detector(model, img_path)
            if hasattr(result, 'pred_instances'):
                counts.append(len(result.pred_instances))
            else:
                # Fallback for older return types
                lanes = result[0] if isinstance(result, list) else result
                counts.append(len(lanes))
        except Exception:
            continue
    return {"avg_lanes": np.mean(counts) if counts else 0, "processed_count": len(counts)}

def main():
    print(f"\n--- Loading Configuration ---")
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.test_pipeline = [
        dict(type='InjectCULaneKeys'),
        dict(type='PackDetInputs')
    ]
        
    print(f"\n--- Loading CLRerNet onto {DEVICE} ---")
    model = init_detector(cfg, CHECKPOINT_PATH, device=DEVICE)

    # FIXED: Convert lists to Tensors for the preprocessor
    if hasattr(model, 'data_preprocessor'):
        model.data_preprocessor.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
        model.data_preprocessor.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)

    all_rows = []
    json_files = glob.glob(os.path.join(LOGS_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files to process.")

    for json_path in json_files:
        filename = Path(json_path).stem
        with open(json_path, 'r') as f:
            try: data = json.load(f)
            except: continue

        for exp in data.get("experiments", []):
            args = exp.get("arguments", {})
            
            # 1. Process Sanity Check
            if "Sanity Check" in exp["data"]:
                sc = exp["data"]["Sanity Check"]
                m = get_lane_stats(model, sc.get("Image Paths", []))
                row = {**args, "run_type": "sanity_check", "run_idx": -1, "seed": "N/A",
                       "avg_lanes_detected": m["avg_lanes"], "images_in_sample": m["processed_count"],
                       "mmd_stat": sc.get("MMD", {}).get("Stat"), "mmd_p_value": "N/A",
                       "mmd_shift_detected": sc.get("MMD", {}).get("Shift Detected")}
                all_rows.append(row)

            # 2. Process Individual Runs
            if "Data Shift Test Data" in exp["data"]:
                runs = exp["data"]["Data Shift Test Data"].get("Individual Test Data", [])
                for run in tqdm(runs, desc=f"Eval {filename}", leave=False):
                    m = get_lane_stats(model, run.get("Image Paths", []))
                    res = run.get("Results", {})
                    row = {**args, "run_type": "test_run", "run_idx": run.get("Run"), "seed": run.get("Seed"),
                           "avg_lanes_detected": m["avg_lanes"], "images_in_sample": m["processed_count"],
                           "mmd_stat": res.get("MMD", {}).get("Stat"), "mmd_p_value": res.get("MMD", {}).get("P-Value"),
                           "mmd_shift_detected": res.get("MMD", {}).get("Shift Detected")}
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
