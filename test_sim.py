import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from loaders import get_dataloader
from models import get_model
from loss.chamfer_loss import ChamferLoss


def test(cfg, checkpoint_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print("Loading test dataset...")
    test_loader = get_dataloader(cfg["data"]["test"])  
    
    
    print("Loading model...")
    model = get_model(cfg["model"]).to(device)
    model.eval()
    
    # load weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
    

    chamfer_loss = ChamferLoss()
    total_loss = 0
    

    print("Running inference on test dataset...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Testing")):
            voxel = data["voxel"].to(device)
            voxel_scale = data["voxel_scale"].to(device)
            gt_pos = data["pos"].to(device)
            gt_ori = data["ori"].to(device)
            gt_param = data["param"].to(device)
            gt_diff_pc = data["diff_pc"].to(device)


            pred_pos, pred_ori, pred_param = model(voxel, voxel_scale)


            loss = chamfer_loss(pred_pos, pred_ori, pred_param, gt_diff_pc)
            total_loss += loss.item()
            
            print(f"Batch {i}: Chamfer Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / len(test_loader)
    print(f"Average Chamfer Loss on test set: {avg_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/voxel_head/voxel_HandlessCup.yml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., results/voxel_HandlessCup/model_best_chamfer_metric.pkl)")
    args = parser.parse_args()


    cfg = OmegaConf.load(args.config)
    

    test(cfg, args.checkpoint)

# python test_sim.py --config configs/voxel_head/voxel_HandlessCup.yml --checkpoint results/voxel_HandlessCup/model_best_chamfer_metric.pkl
# pretrained/voxel/HandlessCup/model_best_chamfer_metric.pkl