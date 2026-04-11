import os
import numpy as np
from tqdm import tqdm, trange
import torch
from models.phase2Autoencoder import ConfP2ConvAutoencoderFC
from models import autoencoderConfigs
import argparse
import concurrent.futures

from utils.mmd_test import mmd_test
from utils.energy_test import energy_test
from utils.bks import bks_distance_test
from utils.mmd_agg import mmdAgg_test
from data.data_builder import get_dataloader, get_seeded_random_dataloader
from data.data_logging import JsonExperimentManager, JsonStyle, JsonDict
import warnings
import torch.multiprocessing as mp
import jax

# This will force a crash if the GPU isn't actually usable
# better to crash and know why than to waste hours on CPU
jax.config.update("jax_platform_name", "gpu")

# Force deterministic behavior for reproducibility
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)


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
        sample_size: int = 1000,
        num_runs: int = 100,
        block_idx: int = 0,
        batch_size: int = 1024,
        image_size: int = 512,
        alpha: float = 0.05,
        seed_base: int = 42,
        file_name: str = "testData.json",
        file_location: str = "./",
        file_style: JsonStyle = 4,
        modelStr: str = "",
        permutation_test_iterations: int = 1000,
        latent_dim: int = 32,
        max_threads: int = None,
    ):
        # Logging Sanity Check Information
        print("Fixed Flag: 4/11/2026")

        # Store all parameters as instance variables for easy access across methods
        self.modelStr = modelStr
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_list_dir = source_list_path
        self.target_list_dir = target_list_path
        self.sample_size = sample_size
        self.num_runs = num_runs
        self.block_idx = block_idx
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_calib = num_runs
        self.alpha = alpha
        self.seed_base = seed_base
        self.permutation_test_iterations = permutation_test_iterations
        self.latent_dim = latent_dim
        self.max_threads = max_threads
        
        # Define all the tests we want to track
        self.test_types = ["MMD", "MMD_Agg", "Energy", "BKS"]

        # Data Storage for Preloaded Features
        self.src_feats = None
        self.calib_data_cache = []
        self.sanity_data_cache = {}
        self.target_data_cache = []

        # Null Stats & Thresholds Tracking
        self.null_stats = {}
        self.tau = {}

        # --- Data Logger ---
        self.datalogger = JsonExperimentManager(
            file_location=file_location, file_name=file_name, style=file_style
        )

        # Log all experiment parameters for reproducibility and analysis
        self.loggerArgs: JsonDict = {
            "CodeMark": "4/11/2026",
            "source_dir": source_dir,
            "target_dir": target_dir,
            "source_list_path": source_list_path,
            "target_list_path": target_list_path,
            "sample_size": sample_size,
            "num_runs": num_runs,
            "block_idx": block_idx,
            "batch_size": batch_size,
            "image_size": image_size,
            "tau_threshold_percentile": 100 * (1 - alpha),
            "seed_base": seed_base,
            "alpha": alpha,
            "deterministic": True,
            "permutation_test_iterations": permutation_test_iterations,
            "latent_dim": latent_dim,
            "test_types": self.test_types
        }

        # This will hold all the experimental results and metadata to be logged at the end
        self.loggerExperimentalData: JsonDict = {}

        # --- GPU Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()
        print(
            f"CUDA Available: {torch.cuda.is_available()} | Total GPUs Found: {num_gpus}"
        )

        # Deterministic behavior for reproducibility
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(42)

        # --- Model Initialization ---
        print("\nInitializing autoencoder...")

        # Select the appropriate model configuration based on the provided modelStr argument
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
            print("Using ASSIST_Taxi pretrained weights for the autoencoder.")
        elif self.modelStr == "DISTILL":
            modelConf = autoencoderConfigs.AutoEncoderWeights.DISTILL
            print("Using Distilled pretrained weights for the autoencoder.")
        else:
            raise ValueError(f"Unsupported model config: {self.modelStr}")

        # Initialize the base model on the appropriate device
        base_model = ConfP2ConvAutoencoderFC(
            configs=modelConf, latent_dim=self.latent_dim
        ).to(self.device)

        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            # Note: DataParallel is used here for simplicity, but for larger models or more complex setups, consider using DistributedDataParallel
            self.model = torch.nn.DataParallel(base_model)
            print(f"Warning: Only {num_gpus} GPUs found. Using all available.")
        else:
            # Even if multiple GPUs are available, we will use only one to avoid potential issues with DataParallel and ensure consistent behavior across different hardware setups.
            self.model = base_model
            print("Using single GPU/CPU.")

    # --- LATENT CACHING HELPER ---
    def _get_or_extract_features(self, loader, list_path, num_samples, seed):
        """
        Checks for cached encodings on disk. If they exist, load them.
        If not, run inference via extract_features() and save them to disk.
        """
        cache_base = "/home1/adoyle2025/Datasets/Encodings"
        # Separate directories by Model String and Latent Dimensionality
        cache_dir = os.path.join(
            cache_base, str(self.modelStr), f"dim_{self.latent_dim}"
        )

        os.makedirs(cache_dir, exist_ok=True)

        # Create a unique, safe filename (e.g., datasets_CULane_list_train_txt_n1000_seed42.npy)
        safe_list_name = (
            os.path.normpath(list_path).replace(os.sep, "_").replace(".", "_")
        )
        file_name = f"{safe_list_name}_n{num_samples}_seed{seed}.npy"
        file_path = os.path.join(cache_dir, file_name)

        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            feats = extract_features(self.model, loader, self.device)
            np.save(file_path, feats)
            return feats

    # --- THREAD WORKER FOR STATISTICAL TESTS ---
    def _execute_test(self, src_feats, tgt_feats, seed):
        mmd_results = mmd_test(
            src_feats, tgt_feats, iterations=self.permutation_test_iterations
        )
        mmdAgg_results = mmdAgg_test(
            src_feats,
            tgt_feats,
            iterations=self.permutation_test_iterations,
            seed=seed,
        )
        energy_results = energy_test(
            src_feats, tgt_feats, iterations=self.permutation_test_iterations
        )
        bks_results = bks_distance_test(src_feats, tgt_feats)
        
        return {
            "MMD": mmd_results,
            "MMD_Agg": mmdAgg_results,
            "Energy": energy_results,
            "BKS": bks_results,
        }

    # STEP 0 — Preload All Features Upfront
    def preload_all_features(self):
        print("\n[STEP 0] Encoding and caching all dataset features...")

        # 0A: Source Features
        loaderReturn = get_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.sample_size,
        )
        self.src_feats = self._get_or_extract_features(
            loaderReturn[0], self.source_list_dir, self.sample_size, "base"
        )

        print(f"{self.source_dir} features loaded. Shape = {self.src_feats.shape}\n")
        self.loggerExperimentalData["Source Training Feature Shape"] = list(
            self.src_feats.shape
        )
        self.loggerExperimentalData["Source Training Feature Image Paths"] = list(
            loaderReturn[1]
        )
        print(f"-> Source features loaded. Shape = {self.src_feats.shape}")

        # 0B: Calibration Features
        for i in tqdm(range(self.num_calib), desc="Preloading Calibration Sets"):
            seed = self.seed_base + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.source_dir,
                list_path=self.source_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.sample_size,
                seed=seed,
            )
            feats = self._get_or_extract_features(
                loaderReturn[0], self.source_list_dir, self.sample_size, seed
            )
            self.calib_data_cache.append((seed, feats, loaderReturn[1]))

        # 0C: Sanity Check Features
        sanity_seed = int(self.seed_base + self.num_calib)
        loaderReturn = get_seeded_random_dataloader(
            root_dir=self.source_dir,
            list_path=self.source_list_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            num_samples=self.sample_size,
            seed=sanity_seed,
            shift=None,
        )
        sanity_feats = self._get_or_extract_features(
            loaderReturn[0], self.source_list_dir, self.sample_size, sanity_seed
        )
        self.sanity_data_cache = {
            "seed": sanity_seed,
            "feats": sanity_feats,
            "paths": loaderReturn[1],
        }
        print("-> Sanity Check features loaded.")

        # 0D: Target Data Shift Features
        for i in tqdm(range(self.num_runs), desc="Preloading Target Sets"):
            seed = self.seed_base + self.num_calib + i
            loaderReturn = get_seeded_random_dataloader(
                root_dir=self.target_dir,
                list_path=self.target_list_dir,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_samples=self.sample_size,
                seed=seed,
            )
            feats = self._get_or_extract_features(
                loaderReturn[0], self.target_list_dir, self.sample_size, seed
            )
            self.target_data_cache.append((seed, feats, loaderReturn[1]))

    # STEP 1 — Calibration (Null Distribution) via Threading
    def calibrate(self):
        calibrationData: JsonDict = {}
        print(f"\n[STEP 1] Calibration using {self.source_dir} (Multithreaded)...")
        calibrationData["Uses"] = self.source_dir

        null_stats_temp = {test: [] for test in self.test_types}
        p_values_temp = {test: [] for test in self.test_types}
        all_image_dirs = {}

        # Dispatch to ThreadPool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_threads
        ) as executor:
            future_to_data = {}
            for seed, feats, img_paths in self.calib_data_cache:
                future = executor.submit(
                    self._execute_test, self.src_feats, feats, seed
                )
                future_to_data[future] = (seed, img_paths)

            for future in tqdm(
                concurrent.futures.as_completed(future_to_data),
                total=self.num_calib,
                desc="Calculating Calibration",
            ):
                seed, img_paths = future_to_data[future]
                all_image_dirs[f"Calibrating with seed {seed}"] = img_paths

                try:
                    result_dict = future.result()
                    for test in self.test_types:
                        t_stat, p_value = result_dict[test]
                        null_stats_temp[test].append(t_stat)
                        if self.permutation_test_iterations > 0:
                            p_values_temp[test].append(float(p_value))
                except Exception as exc:
                    print(f"Calibration generated an exception for seed {seed}: {exc}")

        calibrationData["Result"] = {}

        # Calculate limits for each test
        for test in self.test_types:
            self.null_stats[test] = np.array(null_stats_temp[test])
            self.tau[test] = np.percentile(self.null_stats[test], 100 * (1 - self.alpha))
            
            print(f"  [{test}] τ({1 - self.alpha:.2f}) = {self.tau[test]:.6f}")

            calibrationData["Result"][test] = {
                "Tau": float(self.tau[test]),
                "Mean": float(self.null_stats[test].mean()),
                "Std": float(self.null_stats[test].std()),
            }

            if self.permutation_test_iterations > 0 and len(p_values_temp[test]) > 0:
                avgPVal = np.array(p_values_temp[test]).mean()
                calibrationData["Result"][test]["Average P-Value"] = float(avgPVal)
                calibrationData["Result"][test]["P-Values"] = p_values_temp[test]

        self.loggerExperimentalData["Calibration"] = calibrationData

    # STEP 2 — Sanity Check
    def sanity_check(self):
        sanityCheckData: JsonDict = {}
        print("\n[STEP 2] Sanity Check...")

        sanityCheckData["Image Paths"] = self.sanity_data_cache["paths"]
        sanity_seed = self.sanity_data_cache["seed"]
        sanity_feats = self.sanity_data_cache["feats"]

        result_dict = self._execute_test(self.src_feats, sanity_feats, sanity_seed)
        sanityCheckData["Results"] = {}

        for test in self.test_types:
            t_stat, p_value = result_dict[test]
            
            print(
                f"[SANITY CHECK] {test}(A={self.source_dir}, B={self.source_dir}) = {t_stat:.6f}, τ = {self.tau[test]:.6f}"
            )

            test_data = {
                "Sanity Check Definition": f"{test}(A={self.source_dir}, B={self.source_dir})",
                "Stat": float(t_stat),
                "Tau": float(self.tau[test]),
            }

            if self.permutation_test_iterations > 0:
                test_data["P-Value"] = float(p_value)

            if t_stat <= self.tau[test]:
                test_data["Shift Detected"] = False
                print(f"No shift detected for {test}.")
            else:
                test_data["Shift Detected"] = True
                print(f"False shift detected for {test}.")
                warnings.warn(
                    f"False shift detected in sanity check - {test} exceeded threshold",
                    UserWarning,
                )
            
            sanityCheckData["Results"][test] = test_data
            
        print("")
        self.loggerExperimentalData["Sanity Check"] = sanityCheckData

    # STEP 3 — Data Shift Test via Threading
    def data_shift_test(self):
        dataShiftTestData: JsonDict = {}
        print(f"[STEP 3] Data Shift Test: {self.source_dir} to {self.target_dir}\n")

        dataShiftTestData["Data Shift Test Definition"] = (
            f"{self.source_dir} to {self.target_dir}"
        )
        dataShiftTestData["Runs"] = self.num_runs

        tpr_lists = {test: [] for test in self.test_types}
        stat_values = {test: [] for test in self.test_types}
        dataShiftTestDataTests: list[JsonDict] = []

        # Dispatch to ThreadPool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_threads
        ) as executor:
            future_to_idx = {}
            for idx, (seed, feats, img_paths) in enumerate(self.target_data_cache):
                future = executor.submit(
                    self._execute_test, self.src_feats, feats, seed
                )
                future_to_idx[future] = (idx, seed, img_paths)

            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=self.num_runs,
                desc="Calculating Shifts",
            ):
                idx, seed, img_paths = future_to_idx[future]
                runData: JsonDict = {"Image Paths": img_paths, "Seed": seed, "Run": int(idx + 1), "Results": {}}

                try:
                    result_dict = future.result()
                    
                    for test in self.test_types:
                        t_stat, p_value = result_dict[test]
                        stat_values[test].append(t_stat)

                        detected: bool = t_stat > self.tau[test]
                        tpr_lists[test].append(int(detected))

                        test_run = {
                            "Stat": float(t_stat),
                            "Shift Detected": bool(detected)
                        }

                        if self.permutation_test_iterations > 0:
                            test_run["P-Value"] = float(p_value)
                            
                        runData["Results"][test] = test_run

                    dataShiftTestDataTests.append(runData)

                except Exception as exc:
                    print(
                        f"Data shift test generated an exception for run {idx+1}: {exc}"
                    )

        dataShiftTestData["Individual Test Data"] = dataShiftTestDataTests
        dataShiftTestData["Summary"] = {}

        print("\n[RESULTS] Data Shift detection summary:")
        for test in self.test_types:
            tpr_result = np.mean(tpr_lists[test])
            mean_stat = np.mean(stat_values[test])
            std_stat = np.std(stat_values[test])
            
            print(f"--- {test} ---")
            print(
                f"  Average Stat: {mean_stat:.6f} ± {std_stat:.6f}"
            )
            print(
                f"  TPR (true positive rate) over {self.num_runs} runs: {tpr_result*100:.2f}%"
            )

            dataShiftTestData["Summary"][test] = {
                "TPR": float(tpr_result * 100),
                "Mean Stat": float(mean_stat),
                "Std Stat": float(std_stat)
            }

        self.loggerExperimentalData["Data Shift Test Data"] = dataShiftTestData

    # RUN EVERYTHING
    def run(self):
        # Step 0: Extract/Load all encodings immediately (pure GPU phase)
        self.preload_all_features()
        # Step 1: Multithreaded Calibration (pure CPU Phase)
        self.calibrate()
        # Step 2: Sanity Check (CPU)
        self.sanity_check()
        # Step 3: Multithreaded Target Shift Tests (CPU)
        self.data_shift_test()

        # Log Data
        self.datalogger.add_experiment(
            arguments=self.loggerArgs, data=self.loggerExperimentalData
        )


if __name__ == "__main__":
    # Prevent JAX from preallocating all GPU memory (important for multi-GPU setups and to avoid OOM issues)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Prevent NCCL Deadlocks
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Set multiprocessing start method to 'spawn' for better compatibility with CUDA and to avoid issues on certain platforms
    mp.set_start_method("spawn", force=True)

    # Command-line argument parsing for flexible experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, type=str)
    parser.add_argument("--target_dir", required=True, type=str)
    parser.add_argument(
        "--source_list_path",
        required=True,
        type=str,
        default="./datasets/CULane/list/train.txt",
    )
    parser.add_argument("--target_list_path", required=True, type=str)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--block_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--permutation_test_iterations", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--modelStr", type=str, default="")
    parser.add_argument(
        "--max_threads", type=int, default=None, help="Max thread pool workers"
    )
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
