import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import importlib.util
import glob
from collections import OrderedDict

# ==============================================================================
# 1. THE REGISTRY PATCH (Must be first to prevent 'already registered' crashes)
# ==============================================================================
try:
    import mmengine.registry.registry as registry_mod

    orig_register_module = registry_mod.Registry.register_module

    def patched_register_module(self, *args, **kwargs):
        kwargs["force"] = True
        return orig_register_module(self, *args, **kwargs)

    registry_mod.Registry.register_module = patched_register_module
    print("🛡️  MMEngine Registry class patched globally.")
except Exception:
    pass

# ==============================================================================
# 2. PATH SETUP (Isolating Project vs. Library)
# ==============================================================================
PROJECT_ROOT = "/home1/adoyle2025/Distribution-Shift-Lane-Perception"
CLRERNET_ROOT = "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if CLRERNET_ROOT not in sys.path:
    sys.path.append(CLRERNET_ROOT)

# ==============================================================================
# 3. SECURE IMPORTS & MMDET INITIALIZATION
# ==============================================================================
from mmengine.registry import MODELS, init_default_scope
from mmengine.config import Config

try:
    # A. Local project components
    from models.phase2Autoencoder import ConfP2ConvAutoencoderFC
    from data.data_builder import get_dataloader

    print("✅ Local project components loaded.")

    # B. Initialize MMDetection Scope
    init_default_scope("mmdet")
    print("✅ MMDetection default scope initialized.")

    # C. Discovery: Trigger CLRerNet registration
    import libs.models as clr_models

    if "CLRerNet" in MODELS.module_dict:
        print("🎯 Success: 'CLRerNet' is now in the MMEngine registry.")
    else:
        print("🔍 'CLRerNet' not found in root. Triggering detector registration...")
        import libs.models.detectors.clrernet

        if "CLRerNet" in MODELS.module_dict:
            print("🎯 Success: 'CLRerNet' registered via sub-module.")
        else:
            from libs.models.detectors.clrernet import CLRerNet

            MODELS.register_module(name="CLRerNet", module=CLRerNet, force=True)
            print("✅ Manual registration complete.")

except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)


def build_net(cfg):
    """Uses MMEngine registry to build the model."""
    return MODELS.build(cfg.model if hasattr(cfg, "model") else cfg)


# ==============================================================================
# 4. TRAINING LOGIC
# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TEACHER INITIALIZATION (CLRerNet) ---
    print(f"Initializing Teacher with config: {args.teacher_config}")
    t_cfg_path = (
        args.teacher_config
        if os.path.isabs(args.teacher_config)
        else os.path.join(CLRERNET_ROOT, args.teacher_config)
    )

    teacher_cfg = Config.fromfile(t_cfg_path)
    teacher_model = build_net(teacher_cfg).to(device)

    # --- ROBUST WEIGHT LOADING ---
    print(f"Loading Teacher weights from: {args.teacher_weights}")
    checkpoint = torch.load(args.teacher_weights, map_location=device)

    # Extract state_dict from MMEngine wrapper keys
    if "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        raw_state_dict = checkpoint["model"]
    else:
        raw_state_dict = checkpoint

    # Strip 'module.' prefix if saved from DataParallel
    new_state_dict = OrderedDict()
    for k, v in raw_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    load_info = teacher_model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Teacher Load Info: {load_info}")

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # --- DATA LOADING ---
    print(f"Loading dataset: {args.dataset_name}...")
    train_loader, _ = get_dataloader(
        root_dir=args.dataset_dir,
        list_path=args.dataset_list,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_samples=args.samples,
        cropImg=args.cropImg,
        block_idx=args.block_idx,
    )

    # --- STUDENT INITIALIZATION ---
    student_model = ConfP2ConvAutoencoderFC().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        student_model = nn.DataParallel(student_model)
        teacher_model = nn.DataParallel(teacher_model)

    optimizer = optim.Adam(student_model.parameters(), lr=args.learning_rate)
    mse_loss = nn.MSELoss()
    distill_criterion = nn.MSELoss()

    # --- RESUME CHECKPOINT LOGIC ---
    save_dir = f"checkpoints/Distillation/{args.dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = 0

    checkpoint_files = glob.glob(
        f"{save_dir}/Distill_AE_{args.dataset_name}_epoch_*.pth"
    )
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Resume detected! Loading: {latest_checkpoint}")
        resume_ckpt = torch.load(latest_checkpoint, map_location=device)

        target_model = (
            student_model.module
            if isinstance(student_model, nn.DataParallel)
            else student_model
        )
        target_model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
    else:
        print("Starting distillation training fresh.")

    # --- TRAINING LOOP ---
    print("🚀 Starting Training Loop...")
    for epoch in range(start_epoch, args.epochs):
        student_model.train()
        epoch_total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for imgs in progress_bar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            # ==========================================
            # A. BULLETPROOF TEACHER EXTRACTION
            # ==========================================
            with torch.no_grad():
                # 1. Ask MMDetection for the feature maps directly (bypasses the prediction head)
                try:
                    if isinstance(teacher_model, nn.DataParallel):
                        teacher_output = teacher_model.module.extract_feat(imgs)
                    else:
                        teacher_output = teacher_model.extract_feat(imgs)
                except AttributeError:
                    # Fallback to standard forward if extract_feat doesn't exist
                    teacher_output = teacher_model(imgs)

                # 2. Hunt recursively for the highest-resolution 4D spatial feature map
                teacher_phat = None

                def get_4d_tensor(data):
                    if isinstance(data, torch.Tensor) and data.dim() == 4:
                        return data
                    elif isinstance(data, (list, tuple)):
                        for item in data:
                            res = get_4d_tensor(item)
                            if res is not None:
                                return res
                    elif isinstance(data, dict):
                        for item in data.values():
                            res = get_4d_tensor(item)
                            if res is not None:
                                return res
                    return None

                teacher_phat = get_4d_tensor(teacher_output)

                if teacher_phat is None:
                    raise RuntimeError(
                        f"Could not find a 4D feature map from the Teacher! Got type: {type(teacher_output)}"
                    )

            # ==========================================
            # B. BULLETPROOF STUDENT EXTRACTION
            # ==========================================
            student_output = student_model(imgs)

            reconstructed = None
            if isinstance(student_output, torch.Tensor):
                reconstructed = student_output
            elif isinstance(student_output, (list, tuple)):
                # Hunt specifically for the 4D image tensor [Batch, Channels, Height, Width]
                for item in student_output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        reconstructed = item
                        break
                # Failsafe fallback
                if reconstructed is None:
                    reconstructed = student_output[0]

            # Failsafe check to prevent interpolation crashes
            if reconstructed.dim() < 4:
                raise RuntimeError(
                    f"Could not find a 4D reconstructed image. Found shape: {reconstructed.shape}"
                )

            # ==========================================
            # C. LOSS & ALIGNMENT
            # ==========================================
            # 1. Ensure spatial dimensions match
            if reconstructed.shape[2:] != teacher_phat.shape[2:]:
                teacher_phat = nn.functional.interpolate(
                    teacher_phat, size=reconstructed.shape[2:], mode="bilinear"
                )

            # 2. Ensure channel dimensions match (Prevents MSE channel crash)
            if reconstructed.shape[1] != teacher_phat.shape[1]:
                min_c = min(reconstructed.shape[1], teacher_phat.shape[1])
                reconstructed_distill = reconstructed[:, :min_c, :, :]
                teacher_distill = teacher_phat[:, :min_c, :, :]
            else:
                reconstructed_distill = reconstructed
                teacher_distill = teacher_phat

            # 3. Calculate Loss
            recon_loss = mse_loss(reconstructed, imgs)
            d_loss = distill_criterion(reconstructed_distill, teacher_distill)
            total_loss = recon_loss + (args.distill_weight * d_loss)

            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            progress_bar.set_postfix(
                {"Loss": f"{total_loss.item():.4f}", "Distill": f"{d_loss.item():.4f}"}
            )

        avg_loss = epoch_total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        save_path = f"{save_dir}/Distill_AE_{args.dataset_name}_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": (
                    student_model.module.state_dict()
                    if isinstance(student_model, nn.DataParallel)
                    else student_model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            save_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distillation Training for Lane Perception"
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_list", type=str, required=True)
    parser.add_argument(
        "--teacher_config",
        type=str,
        default="configs/clrernet/culane/clrernet_culane_dla34.py",
    )
    parser.add_argument("--teacher_weights", type=str, required=True)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--cropImg", action="store_true")
    parser.add_argument("--block_idx", type=int, default=0)

    args = parser.parse_args()
    train(args)
