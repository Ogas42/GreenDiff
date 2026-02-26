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
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_linear_from_obs, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import aggregate_sublattice_ldos, flatten_sub_for_energy_ops, g_obs_to_canonical_view, is_sublattice_resolved

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _to_device_tree(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    return x

def find_latest_ckpt(runs_root: str, current_ckpt_dir: str, pattern: str):
    if os.path.exists(current_ckpt_dir):
        ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            return ckpts[-1]
    latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
    if latest_dir:
        ckpts = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            print(f"Found {pattern} in latest run: {latest_dir}")
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
    parser = argparse.ArgumentParser(description="Latent Green Evaluation")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    force_linear_ldos_mode(config, verbose=True, context="test_latent_green")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_dataset_root = os.path.join(project_root, "gf", "data_cache")
    dataset_root = config["paths"].get("dataset_root", "")
    has_cache_index = os.path.exists(os.path.join(dataset_root, "val_index.yaml"))
    has_fallback_index = os.path.exists(os.path.join(fallback_dataset_root, "val_index.yaml"))
    if not has_cache_index and has_fallback_index:
        config["paths"]["dataset_root"] = fallback_dataset_root

    device = torch.device(config["project"]["device"])
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    if args.ckpt_dir:
        current_ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(current_ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        config["paths"]["checkpoints"] = current_ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)

    print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

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

    torch.manual_seed(config["project"]["seed"])
    dataset = GFDataset(config, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    data_cfg = config["data"]
    ldos_cfg = data_cfg.get("ldos_transform", {})
    log_cfg = ldos_cfg.get("log", {})
    log_enabled = ldos_cfg.get("enabled", False) and log_cfg.get("enabled", False)
    log_eps = float(log_cfg.get("eps", 1.0e-6))

    model_cfg = config["latent_green"]["model"]
    sublattice_resolved = bool(is_sublattice_resolved(config))
    loss_type = model_cfg.get("loss_type", "mse")
    huber_beta = float(model_cfg.get("huber_beta", 0.1))
    use_per_energy_affine = bool(model_cfg.get("per_energy_affine", False))
    align_cfg = model_cfg.get("energy_align", {})
    align_enabled = bool(align_cfg.get("enabled", False))
    align_max_shift = int(align_cfg.get("max_shift", 0))
    log_cosh_eps = float(model_cfg.get("log_cosh_eps", 1.0e-6))

    metrics = {
        "mse_model": [],
        "rel_model": [],
        "mse_phys": [],
        "rel_phys": [],
        "mse_phys_affine": [],
        "rel_phys_affine": [],
        "mse_phys_scaled": [],
        "rel_phys_scaled": [],
        "scale_factor": [],
        "mean_ratio": [],
        "peak_ratio": [],
        "pred_p99_over_obs_p99": [],
        "pred_std_over_obs_std": [],
        "psd_error": [],
        "residual": [],
        "hopping_mean": [],
        "hopping_std": [],
    }

    ranges = {
        "pred_lin_min": [],
        "pred_lin_max": [],
        "pred_log_min": [],
        "pred_log_max": [],
        "obs_log_min": [],
        "obs_log_max": [],
        "obs_lin_min": [],
        "obs_lin_max": [],
    }

    print("\nStarting Evaluation...")
    pbar = tqdm(loader, total=min(len(loader), args.num_batches))

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            if idx >= args.num_batches:
                break

            V = batch["V"].to(device)
            g_obs = batch["g_obs"].to(device)
            physics_meta = _to_device_tree(batch.get("physics_meta"), device) if isinstance(batch, dict) else None
            defect_meta = _to_device_tree(batch.get("defect_meta"), device) if isinstance(batch, dict) else None
            z, _, _ = vae.encode(V)

            t_zeros = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
            out = latent_green(z, t_zeros, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True)
            if isinstance(out, tuple):
                g_pred, psi_real, psi_imag, src = out
            else:
                g_pred = out
                psi_real, psi_imag, src = None, None, None

            if sublattice_resolved:
                g_pred_model = flatten_sub_for_energy_ops(g_obs_to_canonical_view(ldos_obs_from_linear(g_pred, data_cfg), data_cfg))
                g_obs_model = flatten_sub_for_energy_ops(g_obs)
                g_pred_lin = aggregate_sublattice_ldos(g_obs_to_canonical_view(g_pred, data_cfg))
                g_obs_lin = aggregate_sublattice_ldos(ldos_linear_from_obs(g_obs, data_cfg))
            else:
                g_pred_lin = g_pred
                g_obs_lin = ldos_linear_from_obs(g_obs, data_cfg)
                g_pred_model = ldos_obs_from_linear(g_pred, data_cfg)
                g_obs_model = g_obs
            if use_per_energy_affine:
                g_pred_model = per_energy_affine(g_pred_model, g_obs_model)
            g_pred_model, _ = align_pred(
                g_pred_model,
                g_obs_model,
                enabled=align_enabled,
                max_shift=align_max_shift,
                loss_type=loss_type,
                huber_beta=huber_beta,
                log_cosh_eps=log_cosh_eps,
            )
            rel_model = torch.norm(g_pred_model - g_obs_model) / torch.norm(g_obs_model).clamp_min(1.0e-6)
            mse_model = F.mse_loss(g_pred_model, g_obs_model).item()
            if log_enabled:
                ranges["pred_log_min"].append(g_pred_model.min().item())
                ranges["pred_log_max"].append(g_pred_model.max().item())
                ranges["obs_log_min"].append(g_obs_model.min().item())
                ranges["obs_log_max"].append(g_obs_model.max().item())

            metrics["mse_model"].append(mse_model)
            metrics["rel_model"].append(rel_model.item())

            pred_lin_raw = g_pred_lin.clamp_min(0)
            obs_lin_raw = g_obs_lin.clamp_min(0)

            metrics["mse_phys"].append(F.mse_loss(pred_lin_raw, obs_lin_raw).item())
            diff_norm_phys = torch.norm(pred_lin_raw - obs_lin_raw, p=2, dim=(1, 2, 3))
            obs_norm_phys = torch.norm(obs_lin_raw, p=2, dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["rel_phys"].append((diff_norm_phys / obs_norm_phys).mean().item())

            pred_lin_eval = pred_lin_raw
            obs_lin_eval = obs_lin_raw
            if use_per_energy_affine:
                pred_lin_eval = per_energy_affine(pred_lin_eval, obs_lin_eval)
            pred_lin_eval, _ = align_pred(
                pred_lin_eval,
                obs_lin_eval,
                enabled=align_enabled,
                max_shift=align_max_shift,
                loss_type=loss_type,
                huber_beta=huber_beta,
                log_cosh_eps=log_cosh_eps,
            )
            metrics["mse_phys_affine"].append(F.mse_loss(pred_lin_eval, obs_lin_eval).item())
            diff_norm_phys_affine = torch.norm(pred_lin_eval - obs_lin_eval, p=2, dim=(1, 2, 3))
            obs_norm_phys_affine = torch.norm(obs_lin_eval, p=2, dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["rel_phys_affine"].append((diff_norm_phys_affine / obs_norm_phys_affine).mean().item())
            
            # Linear Scale Correction (Global Scalar)
            # Find alpha that minimizes || alpha * pred - obs ||^2
            # alpha = (pred . obs) / (pred . pred)
            pred_flat = pred_lin_raw.view(pred_lin_raw.shape[0], -1)
            obs_flat = obs_lin_raw.view(obs_lin_raw.shape[0], -1)
            alpha = (pred_flat * obs_flat).sum(dim=1) / (pred_flat * pred_flat).sum(dim=1).clamp_min(1.0e-8)
            alpha = alpha.view(-1, 1, 1, 1)
            pred_lin_scaled = alpha * pred_lin_raw
            metrics["mse_phys_scaled"].append(F.mse_loss(pred_lin_scaled, obs_lin_raw).item())
            diff_norm_phys_scaled = torch.norm(pred_lin_scaled - obs_lin_raw, p=2, dim=(1, 2, 3))
            metrics["rel_phys_scaled"].append((diff_norm_phys_scaled / obs_norm_phys).mean().item())
            metrics["scale_factor"].append(alpha.mean().item())

            mean_ratio = pred_lin_raw.mean(dim=(1, 2, 3)) / obs_lin_raw.mean(dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["mean_ratio"].append(mean_ratio.mean().item())
            peak_ratio = pred_lin_raw.amax(dim=(1, 2, 3)) / obs_lin_raw.amax(dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["peak_ratio"].extend(peak_ratio.detach().cpu().tolist())
            pred_flat_q = pred_lin_raw.reshape(pred_lin_raw.shape[0], -1)
            obs_flat_q = obs_lin_raw.reshape(obs_lin_raw.shape[0], -1)
            p99_ratio = torch.quantile(pred_flat_q, 0.99, dim=1) / torch.quantile(obs_flat_q, 0.99, dim=1).clamp_min(1.0e-6)
            metrics["pred_p99_over_obs_p99"].extend(p99_ratio.detach().cpu().tolist())
            std_ratio = pred_lin_raw.std(dim=(1, 2, 3)) / obs_lin_raw.std(dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["pred_std_over_obs_std"].extend(std_ratio.detach().cpu().tolist())

            pred_fft = torch.fft.rfft2(pred_lin_raw, norm="ortho")
            obs_fft = torch.fft.rfft2(obs_lin_raw, norm="ortho")
            pred_psd = pred_fft.abs() ** 2
            obs_psd = obs_fft.abs() ** 2
            psd_err = F.l1_loss(torch.log(pred_psd + 1.0e-8), torch.log(obs_psd + 1.0e-8)).item()
            metrics["psd_error"].append(psd_err)

            if psi_real is not None and psi_imag is not None and src is not None:
                res_val = latent_green.residual_loss(
                    psi_real, psi_imag, src, V, physics_meta=physics_meta, defect_meta=defect_meta
                )
                metrics["residual"].append(res_val.mean().item())
            if isinstance(physics_meta, dict) and "hopping" in physics_meta:
                hop = physics_meta["hopping"].detach().float()
                metrics["hopping_mean"].append(hop.mean().item())
                metrics["hopping_std"].append(hop.std().item() if hop.numel() > 1 else 0.0)

            ranges["pred_lin_min"].append(pred_lin_raw.min().item())
            ranges["pred_lin_max"].append(pred_lin_raw.max().item())
            ranges["obs_lin_min"].append(obs_lin_raw.min().item())
            ranges["obs_lin_max"].append(obs_lin_raw.max().item())

            pbar.set_postfix({
                "Rel(Mod)": f"{metrics['rel_model'][-1]:.4f}",
                "Rel(Phys)": f"{metrics['rel_phys'][-1]:.4f}"
            })

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    def print_stat(name, key, fmt=".6f"):
        if metrics[key]:
            vals = metrics[key]
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"{name:<30}: {mean:{fmt}} 卤 {std:{fmt}}")

    print_stat("MSE (Model Space)", "mse_model")
    print_stat("Rel L2 (Model Space)", "rel_model")
    print("-" * 60)
    print_stat("MSE (Physical/Linear)", "mse_phys", ".2e")
    print_stat("Rel L2 (Physical/Linear)", "rel_phys")
    print_stat("MSE (Physical/Linear, Affine)", "mse_phys_affine", ".2e")
    print_stat("Rel L2 (Physical/Linear, Affine)", "rel_phys_affine")
    print_stat("MSE (Physical/Linear, Scaled)", "mse_phys_scaled", ".2e")
    print_stat("Rel L2 (Physical/Linear, Scaled)", "rel_phys_scaled")
    print_stat("Optimal Scale Factor", "scale_factor")
    print_stat("Mean Ratio (Pred/Obs, Linear)", "mean_ratio")
    print_stat("Peak Ratio (Pred/Obs, Linear)", "peak_ratio")
    print_stat("P99 Ratio (Pred/Obs, Linear)", "pred_p99_over_obs_p99")
    print_stat("Std Ratio (Pred/Obs, Linear)", "pred_std_over_obs_std")
    print("-" * 60)
    print_stat("PSD Error (Texture)", "psd_error")
    print_stat("Physical Residual", "residual", ".2e")
    print_stat("Hopping Mean", "hopping_mean")
    print_stat("Hopping Std", "hopping_std")
    print("=" * 60)

    print("\n[Ranges]")
    print(f"Pred Lin: [{np.min(ranges['pred_lin_min']):.6f}, {np.max(ranges['pred_lin_max']):.6f}]")
    print(f"Obs Lin:  [{np.min(ranges['obs_lin_min']):.6f}, {np.max(ranges['obs_lin_max']):.6f}]")
    if log_enabled:
        print(f"Pred Log: [{np.min(ranges['pred_log_min']):.6f}, {np.max(ranges['pred_log_max']):.6f}]")
        print(f"Obs Log:  [{np.min(ranges['obs_log_min']):.6f}, {np.max(ranges['obs_log_max']):.6f}]")
    print(f"Log Enabled: {log_enabled}")

if __name__ == "__main__":
    main()

