#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fake Experiment Runner for Testing FastAPI Wrapper"
    )

    # Core Arguments
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--source_list_path", default="./datasets/CULane/list/train.txt")
    parser.add_argument("--target_list_path", required=True)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--permutation_test_iterations", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--file_location", default="logs")
    parser.add_argument("--file_name", default="experiment.json")

    # Data Perturbations
    parser.add_argument("--gaussian_sigma", type=float, default=0.0)
    parser.add_argument("--rotation_angle", type=float, default=0.0)
    parser.add_argument("--width_shift_frac", type=float, default=0.0)
    parser.add_argument("--height_shift_frac", type=float, default=0.0)
    parser.add_argument("--shear_angle", type=float, default=0.0)
    parser.add_argument("--zoom_factor", type=float, default=1.0)
    parser.add_argument("--crop_image", action="store_true")
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--vertical_flip", action="store_true")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("imagenet_weights")
    subparsers.add_parser("random_weights")
    custom = subparsers.add_parser("custom_weights")
    custom.add_argument("--model_weights_path", required=False)

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed_base)

    print("Initializing autoencoder...")
    print(f"[Autoencoder] - Latent Dim: {args.latent_dim}")
    print(f"Weight Mode: {args.command}")
    print("Using CPU (Mock Execution).")

    # Simulate feature extraction step
    print("\n[STEP 0] Encoding and caching simulated features...")
    print(f"[INFO] Source: {args.source_dir} ({args.sample_size} samples)")
    print(f"[INFO] Target: {args.target_dir} ({args.sample_size} samples)")

    # Simulate calibration step
    print("\n[STEP 1] Calculating Calibration Thresholds (JAX)...")

    tau_thresholds = {
        "MMD": round(random.uniform(0.015, 0.025), 6),
        "Energy": round(random.uniform(0.040, 0.060), 6),
        "BKS": round(random.uniform(0.100, 0.150), 6),
    }

    # Determine if perturbations were applied to adjust synthetic shift detection rate
    has_perturbation = any(
        [
            args.gaussian_sigma > 0,
            args.crop_image,
            args.horizontal_flip,
            args.vertical_flip,
            args.rotation_angle != 0,
            args.width_shift_frac != 0,
            args.height_shift_frac != 0,
            args.shear_angle != 0,
            args.zoom_factor != 1.0,
        ]
    )

    print("\n[STEP 2] Running Shift Detection across runs...")

    runs_data = []
    mmd_hits = 0
    energy_hits = 0

    for run_id in range(1, args.num_runs + 1):
        if has_perturbation:
            mmd_val = tau_thresholds["MMD"] + random.uniform(0.01, 0.05)
            energy_val = tau_thresholds["Energy"] + random.uniform(0.02, 0.08)
        else:
            mmd_val = tau_thresholds["MMD"] + random.uniform(-0.01, 0.005)
            energy_val = tau_thresholds["Energy"] + random.uniform(-0.02, 0.01)

        mmd_shift = mmd_val > tau_thresholds["MMD"]
        energy_shift = energy_val > tau_thresholds["Energy"]

        if mmd_shift:
            mmd_hits += 1
        if energy_shift:
            energy_hits += 1

        runs_data.append(
            {
                "run_id": run_id,
                "MMD": {"statistic": round(mmd_val, 6), "shift_detected": mmd_shift},
                "Energy": {
                    "statistic": round(energy_val, 6),
                    "shift_detected": energy_shift,
                },
            }
        )

    # Build response payload
    results_payload = {
        "config": vars(args),
        "calibration_tau": tau_thresholds,
        "summary_tpr": {
            "MMD": round(mmd_hits / args.num_runs, 4),
            "Energy": round(energy_hits / args.num_runs, 4),
        },
        "individual_test_data": runs_data,
    }

    # Save to target directory
    os.makedirs(args.file_location, exist_ok=True)
    output_path = os.path.join(args.file_location, args.file_name)

    with open(output_path, "w") as f:
        json.dump(results_payload, f, indent=4)

    print(f"\n[SUCCESS] Mock experiment results written to {output_path}")


if __name__ == "__main__":
    main()
