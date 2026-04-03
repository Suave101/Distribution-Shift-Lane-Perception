import os
import numpy as np
from tqdm import trange
import torch
from models.phase2Autoencoder import ConfP2ConvAutoencoderFC
from models import autoencoderConfigs
import argparse

from utils.mmd_test import mmd_test
from data.data_builder import get_dataloader, get_seeded_random_dataloader
from data.data_logging import JsonExperimentManager, JsonStyle, JsonDict
import warnings


# ---------Feature extraction---------
def extract_features(model, loader, device):
    model.eval()
    feats = []

    # Check if we are using DataParallel
    is_parallel = isinstance(model, torch.nn.DataParallel)

    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)

            # if is_parallel:
            #     z = model.encode(imgs)
            # else:
            z = model(imgs, return_encoding=True)

            if z.dim() > 2:
                raise ValueError("Images are still in the pixel space")
                z = z.view(
                    z.size(0), -1
                )  # code to run on raw images (to flatten the image and do the tests)

            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


class ShiftExperiment:
    def __init__(
        self,
        source_dir: str = "./datasets/CULane",
        target_dir: str = "./datasets/CULane",
        source_list_path: str = "./datasets/CULane/list/train.txt",
        target_list_path: str = "./datasets/CULane/list/test.txt",
        src_samples: int = 1000,
        tgt_samples: int = 1000,
        num_runs: int = 10,
        block_idx: int = 0,
        batch_size: int = 1024,
        image_size: int = 512,
        num_calib: int = 100,
        alpha: float = 0.05,
        seed_base: int = 42,
        file_name: str = "testData.json",
        file_location: str = "./",
        file_style: JsonStyle = 4,
        save_all_image_paths: bool = False,
        modelStr: str = ""
    ):
        print("Fixed Flag: 4/1/26")
        self.modelStr = modelStr
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_list_dir = source_list_path
        self.target_list_dir = target_list_path
        self.src_samples = src_samples
        self.tgt_samples = tgt_samples
        self.num_runs = num_runs
        self.block_idx = block_idx
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_calib = num_calib
        self.alpha = alpha
        self.seed_base = seed_base
        self.save_all_image_paths = save_all_image_paths

        # --- Data Logger ---
        self.datalogger = JsonExperimentManager(
            file_location=file_location, file_name=file_name, style=file_style
        )

        self.loggerArgs: JsonDict = {
            "CodeMark": "4/1/2026",
            "source_dir": source_dir,
            "target_dir": target_dir,
            "source_list_path": source_list_path,
            "target_list_path": target_list_path,
            "src_samples": src_samples,
            "tgt_samples": tgt_samples,
            "num_runs": num_runs,
            "block_idx": block_idx,
            "batch_size": batch_size,
            "image_size": image_size,
            "num_calib": num_calib,
            "tau_threshold_percentile": 100 * (1 - alpha),
            "seed_base": seed_base,
            "alpha": alpha,
        }

        self.loggerExperimentalData: JsonDict = {}

        # --- GPU Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print(f"CUDA Available: {torch.cuda.is_available()} | Total GPUs Found: {num_gpus}")
        
        # --- Model Initialization ---
        print("\nInitializing autoencoder...")

        if self.modelStr == "ImageNet":
            modelConf = autoencoderConfigs.AutoEncoderWeights.IMAGE_NET
            print("Using ImageNet pretrained weights for the autoencoder.")
        elif self.modelStr == "Random":
            modelConf = autoencoderConfigs.AutoEncoderWeights.RANDOM_WEIGHTS
            print("Using random weights (untrained) for the autoencoder.")
        elif self.modelStr == "CU_Lane":
            modelConf = autoencoderConfigs.AutoEncoderWeights.CU_LANE
            print("Using CU Lane pretrained weights for the autoencoder.")
        elif self.modelStr == "CurveLanes":
            modelConf = autoencoderConfigs.AutoEncoderWeights.CURVELANES
            print("Using CurveLanes pretrained weights for the autoencoder.")
        elif self.modelStr == "ASSIST_Taxi":
            modelConf = autoencoderConfigs.AutoEncoderWeights.ASSIST_TAXI
            print("Using ASSIST-Taxi pretrained weights for the autoencoder.")
        else:
            raise ValueError(f"Unsupported model config: {self.modelStr}")
        
        base_model = ConfP2ConvAutoencoderFC(configs=modelConf).to(self.device)

        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(base_model)
            print(f"Warning: Only {num_gpus} GPUs found. Using all available.")
        else:
            self.model = base_model
            print("Using single GPU/CPU.")

    # STEP 0 — Load Source Features
    def load_source_features(self):
        loaderReturn = get_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.src_samples,
        )
        loader = loaderReturn[0]
        image_paths = loaderReturn[1]
        self.src_feats = extract_features(self.model, loader, self.device)
        print(f"{self.source_dir} features loaded. Shape = {self.src_feats.shape}\n")
        self.loggerExperimentalData["Source Training Feature Shape"] = list(
            self.src_feats.shape
        )
        self.loggerExperimentalData["Source Training Feature Image Paths"] = list(
            image_paths
        )

    # STEP 1 — Calibration (Null Distribution)
    def calibrate(self):
        calibrationData: JsonDict = {}
        print(f"[STEP 1] Calibration using {self.source_dir}...")
        calibrationData["Uses"] = self.source_dir
        null_stats = []
        all_image_dirs = {}

        for i in trange(self.num_calib, desc="Calibrating"):
            seed = self.seed_base + i
            
            dataloaderReturn = get_seeded_random_dataloader(
                root_dir=self.source_dir,
                list_path=self.source_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.tgt_samples,
                seed=seed,
            )

            calib_src_test_loader = dataloaderReturn[0]

            all_image_dirs[f"Calibrating with seed {seed}"] = dataloaderReturn[1]

            calib_src_test_feats = extract_features(
                self.model, calib_src_test_loader, self.device
            )

            t_stat = mmd_test(self.src_feats, calib_src_test_feats)
            null_stats.append(t_stat)

        self.null_stats = np.array(null_stats)
        self.tau = np.percentile(self.null_stats, 100 * (1 - self.alpha))

        print(f"\n[RESULT] τ({1 - self.alpha:.2f}) = {self.tau:.6f}")
        print(
            f"Mean MMD (same-distribution): {self.null_stats.mean():.6f} ± {self.null_stats.std():.6f}\n"
        )
        calibrationData["Result"] = {
            "Tau": float(self.tau),
            "Mean MMD": float(self.null_stats.mean()),
            "MMD (std)": float(self.null_stats.std()),
        }
        self.loggerExperimentalData["Calibration"] = calibrationData

    # STEP 2 — Sanity Check
    def sanity_check(self):
        sanityCheckData: JsonDict = {}
        print("[STEP 2] Sanity Check...")

        loaderReturn = get_seeded_random_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.tgt_samples,
            seed=int(self.seed_base + self.num_calib),
            shift=None,
        )

        sanity_src_loader = loaderReturn[0]
        sanityCheckData["Image Paths"] = loaderReturn[1]

        sanity_src_feats = extract_features(self.model, sanity_src_loader, self.device)

        mmd_val = mmd_test(self.src_feats, sanity_src_feats)

        print(
            f"[SANITY CHECK] MMD(A = {self.source_dir}, B = {self.source_dir}) = {mmd_val:.6f}, τ = {self.tau:.6f}"
        )
        sanityCheckData["Results"] = {
            "Sanity Check Definition": f"MMD(A = {self.source_dir}, B = {self.source_dir})",
            "MMD": float(mmd_val),
            "Tau": float(self.tau),
        }

        if mmd_val <= self.tau:
            sanityCheckData["Shift Detected"] = bool(False)
            print("No shift detected.\n")
        else:
            sanityCheckData["Shift Detected"] = bool(True)
            print("False shift detected.\n")
            warnings.warn("False shift detected in sanity check - MMD exceeded threshold", UserWarning)

        self.loggerExperimentalData["Sanity Check"] = sanityCheckData

    # STEP 3 — Data Shift Test
    def data_shift_test(self):
        dataShiftTestData: JsonDict = {}
        print(
            f"[STEP 3] Data Shift Test: {self.source_dir} to {self.target_dir}a\n"
        )
        dataShiftTestData["Data Shift Test Definition"] = (
            f"{self.source_dir} to {self.target_dir}"
        )

        dataShiftTestData["Runs"] = self.num_runs

        tpr_list = []
        mmd_values = []
        dataShiftTestDataTests: list[JsonDict] = []

        for i in trange(self.num_runs, desc="Shift Testing"):
            testData: JsonDict = {}
            seed = self.seed_base + self.num_calib + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.target_dir,
                list_path=self.target_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.tgt_samples,
                seed=seed,
            )
            tgt_loader_cross = loaderReturn[0]
            testData["Image Paths"] = loaderReturn[1]
            testData["Seed"] = seed
            tgt_feats_cross = extract_features(
                self.model, tgt_loader_cross, self.device
            )
            mmd_cross = mmd_test(self.src_feats, tgt_feats_cross)

            mmd_values.append(mmd_cross)
            detected: bool = mmd_cross > self.tau
            tpr_list.append(int(detected))

            print(f"[RUN {i+1}] MMD={mmd_cross:.6f} {'✅ Detected' if detected else '❌ Not Detected'}")
            testData["Run"] = int(i + 1)
            testData["MMD"] = float(mmd_cross)
            testData["Shift Detected"] = bool(detected)

            if self.save_all_image_paths:
                dataShiftTestDataTests.append(testData)
            elif i == 0:
                dataShiftTestDataTests.append(testData)

        dataShiftTestData["Individual Test Data"] = dataShiftTestDataTests

        tpr_result = np.mean(tpr_list)
        print("\n[RESULTS] Data Shift detection summary")
        print(f"Average MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
        print(
            f"TPR (true positive rate) over {self.num_runs} runs: {tpr_result*100:.2f}%"
        )
        dataShiftTestData["TPR"] = float(tpr_result * 100)
        dataShiftTestData["Step (c) Mean MMD"] = float(np.mean(mmd_values))
        dataShiftTestData["Step (c) Mean MMD (std)"] = float(np.std(mmd_values))
        self.loggerExperimentalData["Data Shift Test Data"] = dataShiftTestData

    # RUN EVERYTHING
    def run(self):
        # Step 0
        self.load_source_features()
        # Step 1
        self.calibrate()
        # Step 2
        self.sanity_check()
        # Step 3
        self.data_shift_test()
        # Log Data
        self.datalogger.add_experiment(
            arguments=self.loggerArgs, data=self.loggerExperimentalData
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", required=True, type=str
    )
    parser.add_argument(
        "--target_dir", required=True, type=str
    )
    parser.add_argument(
        "--source_list_path",
        required=True,
        type=str,
        default="./datasets/CULane/list/train.txt",
    )
    parser.add_argument(
        "--target_list_path",
        required=True,
        type=str
    )
    parser.add_argument("--src_samples", type=int, default=1000)
    parser.add_argument("--tgt_samples", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_calib", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--save_all_image_paths", type=bool, default=False)
    parser.add_argument("--modelStr", type=str, default="")
    parser.add_argument(
        "--file_location",
        type=str,
        default="logsFixed",
        help="Directory to save the log file.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="sanity_check.json",
        help="Name of the log file.",
    )

    args = parser.parse_args()

    ShiftExperiment(**vars(args)).run()
