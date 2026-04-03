import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from models import phase2Autoencoder
from models import autoencoderConfigs

from data.data_builder import get_dataloader

# ==========================================
# 3. Evaluation Function
# ==========================================
def evaluate_and_reconstruct(model, dataloader, dataset_name, model_name, device, output_dir="./reconstructions32", num_samples=20):
    print(f"\n[*] Evaluating {model_name} on {dataset_name}...\n")
    model.eval()
    mse_loss_fn = nn.MSELoss()
    total_mse = 0.0
    total_samples = 0
    os.makedirs(output_dir, exist_ok=True)
    saved_images = False

    with torch.no_grad():
        for images in tqdm(dataloader, desc=f"Evaluating {model_name} on {dataset_name}"):
            images = images.to(device)
            reconstructions = model(images)
            print(f"Batch MSE: {mse_loss_fn(reconstructions, images).item():.6f}")
            loss = mse_loss_fn(reconstructions, images)
            total_mse += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            if not saved_images:
                n = min(num_samples, images.size(0))
                comparison = torch.cat([images[:n], reconstructions[:n]])
                save_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_reconstructions.png")
                save_image(comparison.cpu(), save_path, nrow=n)
                print("[Model Evaluator] - Images Saved")
                saved_images = True
                print("[Model Evaluator] - MSE Evaluation Has Begun")

    avg_mse = total_mse / total_samples
    print(f"\n[*] Final MSE for {model_name} on {dataset_name}: {avg_mse:.6f}\n")
    return avg_mse

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['curvelanes', 'culane'])
    parser.add_argument('--model_name', type=str, required=True, 
                        choices=["ImageNet", "Random", "CULane", "Curvelanes"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ])
    
    # Configure the requested dataset
    if args.dataset == 'curvelanes':
        root_dir = "/home1/adoyle2025/Datasets/Datasets/Curvelanes" 
        txt_path = "/home1/adoyle2025/Datasets/Datasets/Curvelanes/train/train.txt"
        dataset_name = "CurveLanes_Train"
    else:
        root_dir = "/home1/adoyle2025/Datasets/Datasets/CULane"
        txt_path = "/home1/adoyle2025/Datasets/Datasets/CULane/list/train.txt"
        dataset_name = "CULane_Train"

    loaderReturn = get_dataloader(
            root_dir=root_dir,
            list_path=txt_path,
            batch_size=20,
            image_size=512,
            num_samples=10000,
        )
    
    dataloader = loaderReturn[0]

    configs_dict = {
        "ImageNet": autoencoderConfigs.AutoEncoderWeights.IMAGE_NET,
        "Random": autoencoderConfigs.AutoEncoderWeights.RANDOM_WEIGHTS,
        "CULane": autoencoderConfigs.AutoEncoderWeights.CU_LANE,
        "Curvelanes": autoencoderConfigs.AutoEncoderWeights.CURVELANES
    }
    
    config = configs_dict[args.model_name]
    
    print(f"--- Starting Evaluation ---")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model_name}")
    print(f"---------------------------")
    
    model = phase2Autoencoder.ConfP2ConvAutoencoderFC(latent_dim=32, configs=config).to(device)

    evaluate_and_reconstruct(model, dataloader, dataset_name, args.model_name, device)
