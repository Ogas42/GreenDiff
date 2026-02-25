import os
import sys
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def _ensure_project_root():
    candidate = os.path.abspath(__file__)
    for _ in range(6):
        candidate = os.path.dirname(candidate)
        gf_dir = os.path.join(candidate, "gf")
        if os.path.isdir(gf_dir) and os.path.isfile(os.path.join(gf_dir, "__init__.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
    fallback = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fallback not in sys.path:
        sys.path.insert(0, fallback)

_ensure_project_root()

from gd.data.dataset import GFDataset
from gd.models.vae import VAE
from gd.models.latent_green import LatentGreen
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config, resolve_config_paths
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_linear_from_obs
from gd.utils.obs_layout import aggregate_sublattice_ldos, g_obs_to_canonical_view, is_sublattice_resolved

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def find_latest_ckpt(runs_root: str, current_ckpt_dir: str, pattern: str):
    # 1. Try explicit current_ckpt_dir
    if os.path.exists(current_ckpt_dir):
        ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            return ckpts[-1]
            
    # 2. Try looking in runs_root for latest timestamp directory
    # runs_root often looks like /path/to/runs
    # Inside are timestamp dirs: 2026-02-21_05-09-23
    if os.path.exists(runs_root):
        subdirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
        # Sort by modification time (newest first)
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        for run_dir in subdirs:
            # Check for checkpoints dir inside run_dir
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpts = sorted(glob.glob(os.path.join(ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
                if ckpts:
                    print(f"Found {pattern} in latest run: {run_dir}")
                    return ckpts[-1]
                    
    # 3. Fallback to utility function (which might do similar things but good to have)
    latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
    if latest_dir:
        ckpts = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            print(f"Found {pattern} in latest run (via util): {latest_dir}")
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

def main():
    parser = argparse.ArgumentParser(description="Latent Green Evaluation (Linear Space)")
    parser.add_argument("--config", default="gf/configs/config_clean_physics.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback to local configs directory
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", os.path.basename(args.config))
        if os.path.exists(candidate):
            config_path = candidate
        else:
            # Last resort
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.yaml")
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    force_linear_ldos_mode(config, verbose=True, context="test_green")

    # Check dataset root
    dataset_root = config.get("paths", {}).get("dataset_root", "")
    if "clean_physics" not in dataset_root and "linear" not in dataset_root:
        print(f"Warning: dataset_root might not be linear/clean: {dataset_root}")

    device = torch.device(config["project"]["device"])
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    # Default checkpoints dir if not in config
    current_ckpt_dir = config["paths"].get("checkpoints", os.path.join(runs_root, "checkpoints"))
    
    if args.ckpt_dir:
        current_ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(current_ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        config["paths"]["checkpoints"] = current_ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)

    print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    # Initialize Models
    try:
        vae = VAE(config).to(device).eval()
        latent_green = LatentGreen(config).to(device).eval()
    except ValueError as e:
        if "latent_downsample must be 4" in str(e):
            config["vae"]["latent_downsample"] = 4
            vae = VAE(config).to(device).eval()
            latent_green = LatentGreen(config).to(device).eval()
        else:
            raise

    # Load Checkpoints
    lg_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "latent_green_step_*.pt")
    if lg_ckpt:
        print(f"Loading Latent Green from {lg_ckpt}")
        try:
            latent_green.load_state_dict(normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True)))
        except RuntimeError as e:
            raise RuntimeError(
                "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the Green stage with schema-v2 cache."
            ) from e
    else:
        print("Error: Latent Green checkpoint is required for evaluation!")
        return

    vae_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "vae_step_*.pt")
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: VAE checkpoint not found. Latent space encoding might be random.")

    # Initialize Dataset
    torch.manual_seed(config["project"]["seed"])
    dataset = GFDataset(config, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    sublattice_resolved = bool(is_sublattice_resolved(config))

    # Metrics Containers
    metrics = {
        "mse": [],
        "mae": [],
        "rel_l2": [],
        "peak_error": [],
        "peak_ratio": [],
        "mean_ratio": [],
    }

    ranges = {
        "pred_max": [],
        "obs_max": [],
        "pred_mean": [],
        "obs_mean": [],
    }

    print("\nStarting Linear Evaluation...")
    pbar = tqdm(enumerate(loader), total=min(len(loader), args.num_batches))

    with torch.no_grad():
        for idx, batch in pbar:
            if idx >= args.num_batches:
                break

            V = batch["V"].to(device)
            g_obs = ldos_linear_from_obs(batch["g_obs"].to(device), config)
            
            # Encode V -> z
            z, _, _ = vae.encode(V)
            t_zeros = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
            
            # Predict Latent Green -> g_pred
            out = latent_green(z, t_zeros, return_psi=True)
            if isinstance(out, tuple):
                g_pred, _, _, _ = out
            else:
                g_pred = out
            
            # Ensure non-negative (Physics Constraint)
            if sublattice_resolved:
                g_pred = aggregate_sublattice_ldos(g_obs_to_canonical_view(g_pred, config))
                g_obs = aggregate_sublattice_ldos(g_obs)
            g_pred = g_pred.clamp_min(0)
            
            # `LatentGreen` already predicts linear-space LDOS. Do not invert observation transforms here.
            
            g_obs = g_obs.clamp_min(0)

            # Compute Metrics
            mse = F.mse_loss(g_pred, g_obs).item()
            mae = F.l1_loss(g_pred, g_obs).item()
            
            diff_norm = torch.norm(g_pred - g_obs, p=2, dim=(1, 2, 3))
            obs_norm = torch.norm(g_obs, p=2, dim=(1, 2, 3)).clamp_min(1.0e-9)
            rel = (diff_norm / obs_norm).mean().item()
            
            # Peak Analysis (Critical for Physics)
            pred_max = g_pred.amax(dim=(1, 2, 3))
            obs_max = g_obs.amax(dim=(1, 2, 3))
            peak_err = (pred_max - obs_max).abs().mean().item()
            peak_ratio = (pred_max / obs_max.clamp_min(1.0e-9)).mean().item()
            
            # Mean Analysis
            pred_mean = g_pred.mean(dim=(1, 2, 3))
            obs_mean = g_obs.mean(dim=(1, 2, 3))
            mean_ratio = (pred_mean / obs_mean.clamp_min(1.0e-9)).mean().item()

            # Record
            metrics["mse"].append(mse)
            metrics["mae"].append(mae)
            metrics["rel_l2"].append(rel)
            metrics["peak_error"].append(peak_err)
            metrics["peak_ratio"].append(peak_ratio)
            metrics["mean_ratio"].append(mean_ratio)

            ranges["pred_max"].extend(pred_max.cpu().tolist())
            ranges["obs_max"].extend(obs_max.cpu().tolist())
            ranges["pred_mean"].extend(pred_mean.cpu().tolist())
            ranges["obs_mean"].extend(obs_mean.cpu().tolist())

            pbar.set_postfix({
                "Rel": f"{rel:.4f}",
                "PeakR": f"{peak_ratio:.4f}"
            })

    # Summary
    print("\n" + "=" * 60)
    print("Linear Evaluation Results")
    print("=" * 60)

    def print_stat(name, key, fmt=".2e"):
        if metrics[key]:
            vals = metrics[key]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"{name:<30}: {mean:{fmt}} 卤 {std:{fmt}}")

    print_stat("MSE (Linear)", "mse")
    print_stat("MAE (Linear)", "mae")
    print_stat("Rel L2 (Linear)", "rel_l2", ".4f")
    print("-" * 60)
    print_stat("Peak Error", "peak_error")
    print_stat("Peak Ratio (Pred/Obs)", "peak_ratio", ".4f")
    print_stat("Mean Ratio (Pred/Obs)", "mean_ratio", ".4f")
    print("-" * 60)
    
    print(f"Avg Pred Max: {np.mean(ranges['pred_max']):.2e}")
    print(f"Avg Obs Max:  {np.mean(ranges['obs_max']):.2e}")
    print(f"Avg Pred Mean: {np.mean(ranges['pred_mean']):.2e}")
    print(f"Avg Obs Mean:  {np.mean(ranges['obs_mean']):.2e}")
    print("=" * 60)

if __name__ == "__main__":
    main()

