import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.utils.config_utils import load_config
from gd.inference.teacher_sampler import TeacherSampler
from gd.data.dataset import GFDataset

def find_peak_pixel(img):
    """Find the coordinates of the maximum value pixel."""
    idx = torch.argmax(img)
    h, w = img.shape
    r = idx // w
    c = idx % w
    return r.item(), c.item()

def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality visualization for GreenDiff")
    parser.add_argument("--config", default="configs/config_clean_physics.yaml", help="Path to config file")
    parser.add_argument("--output", default="paper_result.png", help="Output image filename")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to plot")
    parser.add_argument("--ckpt_dir", help="Optional checkpoint directory override")
    args = parser.parse_args()

    # Load Config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        # Try finding it in project root
        alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
        if os.path.exists(alt_path):
            args.config = alt_path
        else:
            return

    config = load_config(args.config)
    if args.ckpt_dir:
        config["paths"]["checkpoints"] = args.ckpt_dir

    device = torch.device(config["project"]["device"])
    
    # Initialize Sampler
    print("Loading models...")
    try:
        sampler = TeacherSampler(config)
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Please ensure you have trained the models first using run_pipeline.py")
        return

    # Load Dataset
    print("Loading test data...")
    dataset = GFDataset(config, split="test")
    
    # Generate Samples
    print(f"Generating {args.num_samples} samples...")
    indices = torch.linspace(0, len(dataset)-1, args.num_samples).long()
    
    # Prepare Plot
    fig = plt.figure(figsize=(20, 5 * args.num_samples))
    gs = GridSpec(args.num_samples, 5, width_ratios=[1, 1, 1, 1, 2])

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        V = sample["V"].unsqueeze(0).to(device)      # (1, H, W)
        g_gt = sample["g_obs"].unsqueeze(0).to(device) # (1, K, H, W)
        
        # Inference
        # Use simple sampling for speed, but high enough steps for quality
        with torch.no_grad():
            g_pred = sampler.sample_diffusion(
                V, 
                steps=config["diffusion"]["sampler"]["steps"],
                eta=config["diffusion"]["sampler"]["eta"],
                guidance_scale=config["guidance"]["lambda"]["lambda0"]
            )
        
        # Select Energy Channel with max activity
        E_idx = torch.argmax(g_gt.mean(dim=(2,3)).squeeze())
        
        v_img = V.squeeze().cpu().numpy()
        gt_img = g_gt[0, E_idx].cpu().numpy()
        pred_img = g_pred[0, E_idx].cpu().numpy()
        
        # Metrics
        mse = np.mean((gt_img - pred_img)**2)
        peak_ratio = pred_img.max() / (gt_img.max() + 1e-6)
        
        # 1. Potential
        ax0 = fig.add_subplot(gs[i, 0])
        im0 = ax0.imshow(v_img, cmap="inferno", origin="lower")
        ax0.set_title(f"Potential V\nSample {idx}")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        ax0.axis("off")
        
        # 2. Ground Truth
        ax1 = fig.add_subplot(gs[i, 1])
        vmin, vmax = gt_img.min(), gt_img.max()
        im1 = ax1.imshow(gt_img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Ground Truth\nEnergy {E_idx}")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis("off")
        
        # 3. Prediction
        ax2 = fig.add_subplot(gs[i, 2])
        im2 = ax2.imshow(pred_img, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Prediction\nPeak Ratio: {peak_ratio:.2f}")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis("off")
        
        # 4. Error Map
        ax3 = fig.add_subplot(gs[i, 3])
        err = np.abs(gt_img - pred_img)
        im3 = ax3.imshow(err, cmap="magma", origin="lower")
        ax3.set_title(f"Abs Error\nMSE: {mse:.2e}")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis("off")
        
        # 5. Line Profile
        ax4 = fig.add_subplot(gs[i, 4])
        # Find peak in GT to choose the line
        pr, pc = find_peak_pixel(g_gt[0, E_idx])
        
        # Horizontal line through peak
        line_gt = gt_img[pr, :]
        line_pred = pred_img[pr, :]
        
        x = np.arange(len(line_gt))
        ax4.plot(x, line_gt, 'k-', linewidth=2, label="Ground Truth")
        ax4.plot(x, line_pred, 'r--', linewidth=2, label="Prediction")
        ax4.set_title(f"Line Profile (Row {pr})")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Mark the line on GT image for reference
        ax1.axhline(pr, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()

