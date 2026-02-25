
import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.data.dataset import GFDataset
from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config, resolve_config_paths
from gd.utils.obs_layout import aggregate_sublattice_ldos

# Fix for OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_checkpoints(teacher, config, device):
    """
    Loads the latest checkpoints for the teacher components.
    """
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    def find_latest_ckpt(pattern):
        # 1. Prefer latest run containing this pattern
        latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
        if latest_dir:
            ckpts = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if ckpts:
                print(f"Found {pattern} in latest run: {latest_dir}")
                return ckpts[-1]
        
        # 2. Fallback to current run
        if os.path.exists(current_ckpt_dir):
            ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if ckpts:
                return ckpts[-1]
        return None

    def normalize_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
            elif k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    # 1. Diffusion
    diff_ckpt = find_latest_ckpt("diffusion_step_*.pt")
    if diff_ckpt:
        print(f"Loading Teacher Diffusion from {diff_ckpt}")
        try:
            teacher.diffusion.load_state_dict(normalize_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True)))
        except RuntimeError as e:
            raise RuntimeError(
                "Diffusion checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the diffusion stage with schema-v2 cache."
            ) from e
    else:
        print("Warning: No Diffusion checkpoint found!")

    # 2. VAE
    vae_ckpt = find_latest_ckpt("vae_step_*.pt")
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        teacher.vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: No VAE checkpoint found!")

    # 3. Latent Green
    lg_ckpt = find_latest_ckpt("latent_green_step_*.pt")
    if lg_ckpt:
        print(f"Loading Latent Green from {lg_ckpt}")
        try:
            teacher.latent_green.load_state_dict(normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True)))
        except RuntimeError as e:
            raise RuntimeError(
                "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the Green stage with schema-v2 cache."
            ) from e
    else:
        print("Warning: No Latent Green checkpoint found!")

def main():
    parser = argparse.ArgumentParser(description="Teacher visualization and diffusion diagnostics")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    args = parser.parse_args()

    print("Starting Teacher Model Visualization...")
    
    # Load config
    config = load_config(args.config)

    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        config["paths"]["checkpoints"] = ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)

    device = torch.device(config["project"]["device"])
    print(f"Using device: {device}")
    
    # Initialize Dataset (Validation split)
    print("Initializing Dataset...")
    dataset = GFDataset(config, split="val")
    # Batch size 4 for visualization
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Force deterministic sampling
    config["diffusion"]["sampler"]["steps"] = 100
    config["diffusion"]["sampler"]["eta"] = 0.0
    config["guidance"]["enabled"] = True
    config["validation"]["enabled"] = False
    config["validation"]["kpm_check"]["enabled"] = False
    config["validation"]["restart"]["enabled"] = False

    # Initialize Teacher
    print("Initializing TeacherSampler...")
    teacher = TeacherSampler(config)
    
    # Load Weights
    load_checkpoints(teacher, config, device)

    # Use trained condition encoder from diffusion
    teacher.condition_encoder = teacher.diffusion.condition_encoder

    # Set to eval mode
    teacher.diffusion.to(device).eval()
    teacher.vae.to(device).eval()
    teacher.condition_encoder.to(device).eval()
    teacher.latent_green.to(device).eval()
    
    # Get a batch
    print("Sampling a batch...")
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Dataset is empty!")
        return

    g_obs = batch["g_obs"].to(device)
    V_true = batch["V"].to(device)
    g_obs_vis = aggregate_sublattice_ldos(g_obs) if g_obs.dim() == 5 else g_obs
    
    print(f"Input shape: {g_obs.shape}")

    with torch.no_grad():
        z, mu, _ = teacher.vae.encode(V_true.unsqueeze(1))
        if mu is not None:
            z = mu
        t = torch.randint(0, teacher.diffusion.T, (z.shape[0],), device=device)
        noise = torch.randn_like(z)
        alpha_t, sigma_t = teacher.diffusion.get_alpha_sigma(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        z_t = alpha_t * z + sigma_t * noise
        pred_eps = teacher.diffusion.predict_eps(z_t, t, g_obs)
        noise_mse = F.mse_loss(pred_eps, noise).item()
    print(f"Noise MSE: {noise_mse:.6f}")
    
    # Generate Prediction
    print("Running Teacher Sampling (this may take a moment)...")
    with torch.no_grad():
        V_pred = teacher.sample(g_obs)
    
    # VAE Reconstruction
    with torch.no_grad():
        V_rec, _, _ = teacher.vae(V_true.unsqueeze(1))

    # Visualization
    print("Visualization...")
    
    n_samples = min(4, g_obs.shape[0])
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]
    
    for i in range(n_samples):
        # Stats
        v_true_min, v_true_max = V_true[i].min().item(), V_true[i].max().item()
        v_rec_min, v_rec_max = V_rec[i, 0].min().item(), V_rec[i, 0].max().item()
        v_pred_min, v_pred_max = V_pred[i, 0].min().item(), V_pred[i, 0].max().item()
        
        print(f"Sample {i}:")
        print(f"  GT   Range: [{v_true_min:.4f}, {v_true_max:.4f}]")
        print(f"  Rec  Range: [{v_rec_min:.4f}, {v_rec_max:.4f}]")
        print(f"  Pred Range: [{v_pred_min:.4f}, {v_pred_max:.4f}]")

        # 1. Input LDOS (First energy channel)
        # g_obs: (B, K, H, W) -> g_obs[i, 0]: (H, W)
        im_in = axes[i, 0].imshow(g_obs_vis[i, 0].cpu().numpy(), cmap="viridis")
        axes[i, 0].set_title("Input LDOS (Agg E0)" if g_obs.dim() == 5 else "Input LDOS (E0)")
        axes[i, 0].axis("off")
        plt.colorbar(im_in, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # 2. Ground Truth Potential
        # V_true: (B, H, W) -> V_true[i]: (H, W)
        im_gt = axes[i, 1].imshow(V_true[i].cpu().numpy(), cmap="inferno")
        axes[i, 1].set_title(f"Ground Truth\n[{v_true_min:.2f}, {v_true_max:.2f}]")
        axes[i, 1].axis("off")
        plt.colorbar(im_gt, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. VAE Reconstruction
        # V_rec: (B, 1, H, W) -> V_rec[i, 0]: (H, W)
        im_rec = axes[i, 2].imshow(V_rec[i, 0].cpu().numpy(), cmap="inferno")
        axes[i, 2].set_title(f"VAE Reconstruction\n[{v_rec_min:.2f}, {v_rec_max:.2f}]")
        axes[i, 2].axis("off")
        plt.colorbar(im_rec, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # 4. Teacher Prediction
        # V_pred: (B, 1, H, W) -> V_pred[i, 0]: (H, W)
        im_pred = axes[i, 3].imshow(V_pred[i, 0].cpu().numpy(), cmap="inferno")
        axes[i, 3].set_title(f"Teacher Prediction\n[{v_pred_min:.2f}, {v_pred_max:.2f}]")
        axes[i, 3].axis("off")
        plt.colorbar(im_pred, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
    save_path = "teacher_vis_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Visualization saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()

