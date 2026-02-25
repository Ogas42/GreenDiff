import os
import sys
import glob
import re
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gd.data.dataset import GFDataset
from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config, resolve_config_paths
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_obs_from_linear

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def _ckpt_step(path):
    base = os.path.basename(path)
    match = re.search(r"_step_(\d+)\.pt$", base)
    if not match:
        return None
    return int(match.group(1))

def find_latest_ckpt(runs_root, current_ckpt_dir, pattern, prefer_ckpt_dir=None):
    if prefer_ckpt_dir and os.path.exists(prefer_ckpt_dir):
        ckpts = [p for p in glob.glob(os.path.join(prefer_ckpt_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            return ckpts[-1]
    latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
    if latest_dir:
        ckpts = [p for p in glob.glob(os.path.join(latest_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            print(f"Found {pattern} in latest run: {latest_dir}")
            return ckpts[-1]
    if os.path.exists(current_ckpt_dir):
        ckpts = [p for p in glob.glob(os.path.join(current_ckpt_dir, pattern)) if _ckpt_step(p) is not None]
        ckpts = sorted(ckpts, key=_ckpt_step)
        if ckpts:
            return ckpts[-1]
    return None

def main():
    parser = argparse.ArgumentParser(description="Diffusion test with visualization and metrics")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    parser.add_argument("--out", default="diffusion_test_result.png", help="Output image path")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    base_config = load_config(args.config)
    config = base_config
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        run_config_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(run_config_path):
            config = load_config(run_config_path)
        config["paths"]["checkpoints"] = ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)

    device = torch.device(config["project"]["device"])
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    if not args.ckpt_dir:
        latest_ckpt_dir = get_latest_checkpoint_dir(runs_root, require_pattern="diffusion_step_*.pt")
        if latest_ckpt_dir:
            run_dir = os.path.dirname(latest_ckpt_dir)
            run_config_path = os.path.join(run_dir, "config.yaml")
            if os.path.exists(run_config_path):
                config = load_config(run_config_path)
                config["paths"]["checkpoints"] = latest_ckpt_dir
                config["paths"]["workdir"] = run_dir
                config["paths"]["runs_root"] = runs_root
                config = resolve_config_paths(config)
                device = torch.device(config["project"]["device"])
                current_ckpt_dir = config["paths"]["checkpoints"]

    force_linear_ldos_mode(config, verbose=True, context="test_diffusion")

    config["diffusion"]["sampler"]["steps"] = 100
    config["diffusion"]["sampler"]["eta"] = 0.0
    config["guidance"]["enabled"] = True
    config["validation"]["enabled"] = False
    config["validation"]["kpm_check"]["enabled"] = False
    config["validation"]["restart"]["enabled"] = False

    print(f"Dataset LDOS Config: {config.get('data', {}).get('ldos_transform')}")

    dataset = GFDataset(config, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    try:
        teacher = TeacherSampler(config)
    except ValueError as e:
        if "latent_downsample must be 4" in str(e):
            config["vae"]["latent_downsample"] = 4
            teacher = TeacherSampler(config)
        else:
            raise

    diff_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "diffusion_step_*.pt")
    if diff_ckpt:
        print(f"Loading Diffusion from {diff_ckpt}")
        state = normalize_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True))
        ema_path = diff_ckpt.replace(".pt", "_ema.pt")
        if os.path.exists(ema_path):
            print(f"Loading Diffusion EMA from {ema_path}")
            state = normalize_state_dict(torch.load(ema_path, map_location=device, weights_only=True))
        teacher.diffusion.load_state_dict(state)
    else:
        print("Error: No Diffusion checkpoint found! Cannot proceed with valid testing.")
        # We will continue for debugging purposes but warn heavily
        print("WARNING: RUNNING WITH RANDOM WEIGHTS - RESULTS WILL BE GARBAGE")

    prefer_ckpt_dir = os.path.dirname(diff_ckpt) if diff_ckpt else None
    vae_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "vae_step_*.pt", prefer_ckpt_dir=prefer_ckpt_dir)
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        teacher.vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: No VAE checkpoint found.")

    lg_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "latent_green_step_*.pt", prefer_ckpt_dir=prefer_ckpt_dir)
    if lg_ckpt:
        print(f"Loading Latent Green from {lg_ckpt}")
        teacher.latent_green.load_state_dict(normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True)))
    else:
        print("Warning: No Latent Green checkpoint found.")

    teacher.condition_encoder = teacher.diffusion.condition_encoder
    teacher.diffusion.to(device).eval()
    teacher.vae.to(device).eval()
    teacher.condition_encoder.to(device).eval()
    teacher.latent_green.to(device).eval()

    batch = next(iter(loader))
    g_obs = batch["g_obs"].to(device)
    V_true = batch["V"].to(device)

    print(f"g_obs range: min={g_obs.min().item():.4f}, max={g_obs.max().item():.4f}, mean={g_obs.mean().item():.4f}")

    # DIAGNOSTIC: Try normalizing g_obs if it looks like log-data (-13) but maybe model expects normalized (0)
    # or if it looks like raw data (1e-6) and model expects log.
    g_obs_norm = g_obs.clone()
    if g_obs.mean() < -5.0: # Likely log transformed
        print("Diagnostic: g_obs seems log-transformed. Creating normalized version for testing.")
        # Normalize per sample (preserving spectral shape across K)
        mean = g_obs.mean(dim=(1, 2, 3), keepdim=True)
        std = g_obs.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        g_obs_norm = (g_obs - mean) / std
        print(f"g_obs_norm range: {g_obs_norm.min().item():.2f} to {g_obs_norm.max().item():.2f}")
    
    with torch.no_grad():
        z, mu, _ = teacher.vae.encode(V_true.unsqueeze(1))
        if mu is not None:
            z = mu

        latent_scale_cfg = config["diffusion"]["training"].get("latent_scale", {})
        latent_scale_mode = latent_scale_cfg.get("mode", "none")
        latent_target_std = float(latent_scale_cfg.get("target_std", 1.0))
        latent_scale = float(latent_scale_cfg.get("scale", 1.0))
        
        if latent_scale_mode == "auto":
            z_std = z.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            z_scaled = z * (latent_target_std / z_std)
            teacher.unscale_factor = z_std / latent_target_std
        elif latent_scale_mode == "fixed":
            z_scaled = z * latent_scale
            teacher.unscale_factor = 1.0 / latent_scale
        else:
            z_scaled = z
            teacher.unscale_factor = 1.0

        t = torch.randint(0, teacher.diffusion.T, (z.shape[0],), device=device)
        noise = torch.randn_like(z_scaled)
        alpha_t, sigma_t = teacher.diffusion.get_alpha_sigma(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        
        z_t = alpha_t * z_scaled + sigma_t * noise
        
        pred_eps = teacher.diffusion.predict_eps(z_t, t, g_obs)
        noise_mse = F.mse_loss(pred_eps, noise).item()
        
        x0_pred = teacher.diffusion.predict_x0(z_t, t, g_obs)
        x0_raw = F.mse_loss(x0_pred, z_scaled).item()

        teacher.guidance_cfg["enabled"] = True
        V_pred = teacher.sample(g_obs)
        teacher.guidance_cfg["enabled"] = False
        V_pred_unguided = teacher.sample(g_obs)
        
        # Try normalized condition
        teacher.guidance_cfg["enabled"] = True
        V_pred_norm = teacher.sample(g_obs_norm)

        teacher.guidance_cfg["enabled"] = True
        V_rec, _, _ = teacher.vae(V_true.unsqueeze(1))
        
        # Phys MSE calculation
        z_pred, mu_pred, _ = teacher.vae.encode(V_pred)
        if mu_pred is not None:
            z_pred = mu_pred
        
        # LatentGreen expects t (time embedding) if trained with it, but usually just z for static prediction
        # Check LatentGreen forward signature
        # In teacher_sampler: g_pred = teacher.latent_green(z_new, t_batch)
        # So we need t. For "final" prediction, what t?
        # LatentGreen is a proxy trained to map z -> g. If it takes t, it might be time-dependent (unlikely for physics proxy)
        # or it ignores t.
        # Let's use t=0 (clean)
        t_zeros = torch.zeros((z_pred.shape[0],), dtype=torch.long, device=z_pred.device)
        g_pred = teacher.latent_green(z_pred, t_zeros)
        
        # Check if we need to log-transform g_pred for comparison
        g_pred = ldos_obs_from_linear(g_pred, config.get("data", {}))
        phys_mse = F.mse_loss(g_pred, g_obs).item()
        
        # Phys MSE for Normalized Cond
        z_pred_norm, mu_pred_norm, _ = teacher.vae.encode(V_pred_norm)
        if mu_pred_norm is not None:
            z_pred_norm = mu_pred_norm
        g_pred_norm = teacher.latent_green(z_pred_norm, t_zeros)
        g_pred_norm = ldos_obs_from_linear(g_pred_norm, config.get("data", {}))
        phys_mse_norm = F.mse_loss(g_pred_norm, g_obs).item() # Compare against original g_obs (as target)

        # Phys MSE for Ground Truth (Control Group)
        z_true, mu_true, _ = teacher.vae.encode(V_true.unsqueeze(1))
        if mu_true is not None:
            z_true = mu_true
        g_pred_true = teacher.latent_green(z_true, t_zeros)
        
        # Stats for debugging
        g_pred_lin_mean = g_pred_true.mean().item()
        g_pred_lin_min = g_pred_true.min().item()
        g_pred_lin_max = g_pred_true.max().item()

        g_pred_true = ldos_obs_from_linear(g_pred_true, config.get("data", {}))
        
        phys_mse_gt = F.mse_loss(g_pred_true, g_obs).item()
        g_pred_obs_mean = g_pred_true.mean().item()

    print(f"Noise MSE: {noise_mse:.6f}")
    print(f"x0 Raw (Latent MSE): {x0_raw:.6f}")
    print(f"Phys MSE (Prediction): {phys_mse:.6f}")
    print(f"Phys MSE (Norm Cond): {phys_mse_norm:.6f}")
    print(f"Phys MSE (Ground Truth): {phys_mse_gt:.6f} <--- If this is high, LatentGreen is broken/mismatched")
    
    print("-" * 40)
    print(f"LatentGreen Output Stats (Linear) on GT: Min={g_pred_lin_min:.2e}, Max={g_pred_lin_max:.2e}, Mean={g_pred_lin_mean:.2e}")
    print(f"LatentGreen Output Stats (Obs) on GT: Mean={g_pred_obs_mean:.4f}")
    print(f"Observed Data Stats (Obs): Mean={g_obs.mean().item():.4f}")
    print("-" * 40)

    print(f"V_pred range: {V_pred.min().item():.2f} to {V_pred.max().item():.2f}")
    print(f"V_true range: {V_true.min().item():.2f} to {V_true.max().item():.2f}")

    # Plotting
    B = g_obs.shape[0]
    fig, axes = plt.subplots(B, 8, figsize=(32, 4 * B))
    
    for i in range(B):
        # Ground Truth
        ax = axes[i, 0]
        im = ax.imshow(V_true[i].cpu().numpy(), cmap="inferno")
        ax.set_title("Ground Truth")
        plt.colorbar(im, ax=ax)
        
        # Prediction
        ax = axes[i, 1]
        im = ax.imshow(V_pred[i, 0].cpu().numpy(), cmap="inferno")
        ax.set_title("Prediction")
        plt.colorbar(im, ax=ax)
        
        # Error
        ax = axes[i, 2]
        err = torch.abs(V_true[i] - V_pred[i, 0])
        im = ax.imshow(err.cpu().numpy(), cmap="magma")
        ax.set_title(f"Abs Error {err.mean().item():.4f}")
        plt.colorbar(im, ax=ax)

        # Prediction (Unguided)
        ax = axes[i, 3]
        im = ax.imshow(V_pred_unguided[i, 0].cpu().numpy(), cmap="inferno")
        ax.set_title("Prediction (unguided)")
        plt.colorbar(im, ax=ax)

        # Prediction (Norm Cond)
        ax = axes[i, 4]
        im = ax.imshow(V_pred_norm[i, 0].cpu().numpy(), cmap="inferno")
        ax.set_title("Prediction (Norm Cond)")
        plt.colorbar(im, ax=ax)
        
        # Error (Norm Cond)
        ax = axes[i, 5]
        err_n = torch.abs(V_true[i] - V_pred_norm[i, 0])
        im = ax.imshow(err_n.cpu().numpy(), cmap="magma")
        ax.set_title(f"Abs Error N {err_n.mean().item():.4f}")
        plt.colorbar(im, ax=ax)
        
        # LDOS (Condition) - show a few channels
        # g_obs is (K, H, W). Let's show channel 0 and K//2
        K = g_obs.shape[1]
        ax = axes[i, 6]
        im = ax.imshow(g_obs[i, 0].cpu().numpy(), cmap="viridis")
        ax.set_title(f"LDOS E0")
        plt.colorbar(im, ax=ax)
        
        ax = axes[i, 7]
        im = ax.imshow(g_obs[i, K//2].cpu().numpy(), cmap="viridis")
        ax.set_title(f"LDOS E{K//2}")
        plt.colorbar(im, ax=ax)


    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    main()

