from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gd.core.checkpoints.manager import CheckpointManager, normalize_state_dict_keys
from gd.core.logging.progress import get_tqdm
from gd.core.logging.results import (
    checkpoint_fingerprint,
    config_fingerprint,
    hardware_info,
    save_eval_result_json,
    utc_timestamp,
)
from gd.data.dataset import GFDataset
from gd.models.latent_green import LatentGreen
from gd.models.vae import VAE
from gd.utils.config_utils import resolve_config_paths
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_linear_from_obs, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import (
    aggregate_sublattice_ldos,
    flatten_sub_for_energy_ops,
    g_obs_to_canonical_view,
    is_sublattice_resolved,
)


def _to_device_tree(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device) for k, v in x.items()}
    return x


def _safe_mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _summarize_metrics(metrics: Dict[str, list[float]]) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}
    for key, values in metrics.items():
        mean, std = _safe_mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    # Canonical aliases for report pipeline
    summary["mse"] = summary.get("mse_phys_mean")
    summary["mae"] = summary.get("mae_phys_mean")
    summary["rel_l2"] = summary.get("rel_phys_mean")
    summary["psd_error"] = summary.get("psd_error_mean")
    summary["residual"] = summary.get("residual_mean")
    return summary


def _prepare_config(config: Dict[str, Any], ckpt_dir: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    force_linear_ldos_mode(cfg, verbose=True, context="gd.eval.green_eval")
    if ckpt_dir:
        run_dir = os.path.dirname(ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        cfg["paths"]["checkpoints"] = ckpt_dir
        cfg["paths"]["workdir"] = run_dir
        cfg["paths"]["runs_root"] = runs_root
        cfg = resolve_config_paths(cfg, cfg.get("paths", {}).get("config_path"))
    return cfg


def _load_models(cfg: Dict[str, Any], device: torch.device, ckpt_dir: Optional[str]) -> tuple[VAE, LatentGreen, str, Optional[str]]:
    runs_root = cfg["paths"]["runs_root"]
    current_ckpt_dir = ckpt_dir or cfg["paths"]["checkpoints"]
    mgr = CheckpointManager(runs_root=runs_root, current_ckpt_dir=current_ckpt_dir)

    try:
        vae = VAE(cfg).to(device).eval()
        latent_green = LatentGreen(cfg).to(device).eval()
    except ValueError as e:
        if "latent_downsample must be 4" in str(e):
            cfg["vae"]["latent_downsample"] = 4
            vae = VAE(cfg).to(device).eval()
            latent_green = LatentGreen(cfg).to(device).eval()
        else:
            raise

    lg_ckpt = mgr.find_latest("latent_green_step_*.pt")
    if not lg_ckpt:
        raise FileNotFoundError("Latent Green checkpoint is required for evaluation.")
    try:
        latent_green.load_state_dict(
            normalize_state_dict_keys(torch.load(lg_ckpt, map_location=device, weights_only=True))
        )
    except RuntimeError as e:
        raise RuntimeError(
            "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
            "Re-train the Latent Green checkpoint on schema-v2 cache."
        ) from e

    vae_ckpt = mgr.find_latest("vae_step_*.pt")
    if vae_ckpt:
        vae.load_state_dict(normalize_state_dict_keys(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("[gd.eval.green_eval] Warning: VAE checkpoint not found; evaluation may be invalid.")

    return vae, latent_green, lg_ckpt, vae_ckpt


def run(
    config: Dict[str, Any],
    runtime_ctx: Any = None,
    ckpt_dir: Optional[str] = None,
    split: str = "val",
    max_batches: Optional[int] = None,
    save_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    dataset_suite: str = "synthetic_main_v1",
    quiet: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    cfg = _prepare_config(config, ckpt_dir)
    device = torch.device(runtime_ctx.dist.device if runtime_ctx is not None else cfg["project"]["device"])

    if batch_size is None:
        batch_size = int(kwargs.get("batch_size", 4))
    if max_batches is None:
        max_batches = int(kwargs.get("num_batches", 50))

    vae, latent_green, lg_ckpt, vae_ckpt = _load_models(cfg, device, ckpt_dir=ckpt_dir)

    torch.manual_seed(int(cfg["project"]["seed"]))
    dataset = GFDataset(cfg, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    data_cfg = cfg["data"]
    ldos_cfg = data_cfg.get("ldos_transform", {})
    log_cfg = ldos_cfg.get("log", {})
    log_enabled = bool(ldos_cfg.get("enabled", False) and log_cfg.get("enabled", False))
    log_eps = float(log_cfg.get("eps", 1.0e-6))

    model_cfg = cfg["latent_green"]["model"]
    loss_type = model_cfg.get("loss_type", "mse")
    huber_beta = float(model_cfg.get("huber_beta", 0.1))
    use_per_energy_affine = bool(model_cfg.get("per_energy_affine", False))
    align_cfg = model_cfg.get("energy_align", {})
    align_enabled = bool(align_cfg.get("enabled", False))
    align_max_shift = int(align_cfg.get("max_shift", 0))
    log_cosh_eps = float(model_cfg.get("log_cosh_eps", 1.0e-6))
    sublattice_resolved = bool(is_sublattice_resolved(data_cfg))

    metrics: Dict[str, list[float]] = {
        "mse_model": [],
        "rel_model": [],
        "mse_phys": [],
        "mae_phys": [],
        "rel_phys": [],
        "mse_phys_sub": [],
        "mae_phys_sub": [],
        "rel_phys_sub": [],
        "mse_phys_affine": [],
        "rel_phys_affine": [],
        "mse_phys_scaled": [],
        "rel_phys_scaled": [],
        "scale_factor": [],
        "mean_ratio": [],
        "psd_error": [],
        "residual": [],
        "hopping_mean": [],
        "hopping_std": [],
    }

    tqdm = get_tqdm()
    total = min(len(loader), max_batches if max_batches is not None else len(loader))
    pbar = tqdm(loader, total=total, disable=quiet)

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            if max_batches is not None and idx >= max_batches:
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
                g_pred_obs_c = g_obs_to_canonical_view(ldos_obs_from_linear(g_pred, data_cfg), data_cfg)
                g_obs_obs_c = g_obs
                g_pred_model = flatten_sub_for_energy_ops(g_pred_obs_c)
                g_obs_model = flatten_sub_for_energy_ops(g_obs_obs_c)
                g_pred_lin_c = g_obs_to_canonical_view(g_pred, data_cfg).clamp_min(0)
                g_obs_lin_c = ldos_linear_from_obs(g_obs, data_cfg).clamp_min(0)
                g_pred_lin_sub = flatten_sub_for_energy_ops(g_pred_lin_c)
                g_obs_lin_sub = flatten_sub_for_energy_ops(g_obs_lin_c)
                g_obs_lin = aggregate_sublattice_ldos(g_obs_lin_c)
                pred_lin_raw = aggregate_sublattice_ldos(g_pred_lin_c).clamp_min(0)
                obs_lin_raw = g_obs_lin.clamp_min(0)
            else:
                g_pred_model = ldos_obs_from_linear(g_pred, data_cfg)
                g_obs_model = g_obs
                g_obs_lin = ldos_linear_from_obs(g_obs, data_cfg).clamp_min(0)
                pred_lin_raw = g_pred.clamp_min(0)
                obs_lin_raw = g_obs_lin
                g_pred_lin_sub = pred_lin_raw
                g_obs_lin_sub = obs_lin_raw

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

            metrics["mse_model"].append(float(mse_model))
            metrics["rel_model"].append(float(rel_model.item()))

            metrics["mse_phys"].append(float(F.mse_loss(pred_lin_raw, obs_lin_raw).item()))
            metrics["mae_phys"].append(float(F.l1_loss(pred_lin_raw, obs_lin_raw).item()))
            diff_norm_phys = torch.norm(pred_lin_raw - obs_lin_raw, p=2, dim=(1, 2, 3))
            obs_norm_phys = torch.norm(obs_lin_raw, p=2, dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["rel_phys"].append(float((diff_norm_phys / obs_norm_phys).mean().item()))
            metrics["mse_phys_sub"].append(float(F.mse_loss(g_pred_lin_sub, g_obs_lin_sub).item()))
            metrics["mae_phys_sub"].append(float(F.l1_loss(g_pred_lin_sub, g_obs_lin_sub).item()))
            diff_norm_phys_sub = torch.norm(g_pred_lin_sub - g_obs_lin_sub, p=2, dim=(1, 2, 3))
            obs_norm_phys_sub = torch.norm(g_obs_lin_sub, p=2, dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["rel_phys_sub"].append(float((diff_norm_phys_sub / obs_norm_phys_sub).mean().item()))

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
            metrics["mse_phys_affine"].append(float(F.mse_loss(pred_lin_eval, obs_lin_eval).item()))
            diff_norm_phys_affine = torch.norm(pred_lin_eval - obs_lin_eval, p=2, dim=(1, 2, 3))
            obs_norm_phys_affine = torch.norm(obs_lin_eval, p=2, dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["rel_phys_affine"].append(float((diff_norm_phys_affine / obs_norm_phys_affine).mean().item()))

            pred_flat = pred_lin_raw.view(pred_lin_raw.shape[0], -1)
            obs_flat = obs_lin_raw.view(obs_lin_raw.shape[0], -1)
            alpha = (pred_flat * obs_flat).sum(dim=1) / (pred_flat * pred_flat).sum(dim=1).clamp_min(1.0e-8)
            alpha = alpha.view(-1, 1, 1, 1)
            pred_lin_scaled = alpha * pred_lin_raw
            metrics["mse_phys_scaled"].append(float(F.mse_loss(pred_lin_scaled, obs_lin_raw).item()))
            diff_norm_phys_scaled = torch.norm(pred_lin_scaled - obs_lin_raw, p=2, dim=(1, 2, 3))
            metrics["rel_phys_scaled"].append(float((diff_norm_phys_scaled / obs_norm_phys).mean().item()))
            metrics["scale_factor"].append(float(alpha.mean().item()))

            mean_ratio = pred_lin_raw.mean(dim=(1, 2, 3)) / obs_lin_raw.mean(dim=(1, 2, 3)).clamp_min(1.0e-6)
            metrics["mean_ratio"].append(float(mean_ratio.mean().item()))

            pred_fft = torch.fft.rfft2(pred_lin_raw, norm="ortho")
            obs_fft = torch.fft.rfft2(obs_lin_raw, norm="ortho")
            pred_psd = pred_fft.abs() ** 2
            obs_psd = obs_fft.abs() ** 2
            metrics["psd_error"].append(float(F.l1_loss(torch.log(pred_psd + 1.0e-8), torch.log(obs_psd + 1.0e-8)).item()))

            if psi_real is not None and psi_imag is not None and src is not None:
                res_val = latent_green.residual_loss(
                    psi_real, psi_imag, src, V, physics_meta=physics_meta, defect_meta=defect_meta
                )
                metrics["residual"].append(float(res_val.mean().item()))
            if isinstance(physics_meta, dict) and "hopping" in physics_meta:
                hop = physics_meta["hopping"].detach().float()
                metrics["hopping_mean"].append(float(hop.mean().item()))
                metrics["hopping_std"].append(float(hop.std().item()) if hop.numel() > 1 else 0.0)

            if not quiet and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    {
                        "Rel": f"{metrics['rel_phys'][-1]:.4f}",
                        "PSD": f"{metrics['psd_error'][-1]:.4f}",
                    }
                )

    summary_metrics = _summarize_metrics(metrics)
    run_id = os.path.basename(cfg["paths"]["workdir"])
    result = {
        "task": "green_eval",
        "dataset_suite": dataset_suite,
        "variant": kwargs.get("variant"),
        "split": split,
        "seed": int(cfg["project"]["seed"]),
        "run_id": run_id,
        "metrics": summary_metrics,
        "artifacts": {},
        "config_hash": config_fingerprint(cfg),
        "checkpoint_tag": os.path.basename(lg_ckpt),
        "checkpoint_hash": checkpoint_fingerprint(lg_ckpt),
        "timestamp": utc_timestamp(),
        "hardware": hardware_info(),
        "meta": {
            "num_batches": max_batches,
            "batch_size": batch_size,
            "log_enabled": log_enabled,
            "vae_checkpoint": os.path.basename(vae_ckpt) if vae_ckpt else None,
        },
    }

    if save_json is None and output_dir:
        save_json = os.path.join(output_dir, f"{run_id}_green_eval_{split}.json")
    if save_json:
        result["artifacts"]["result_json"] = save_json
        path = save_eval_result_json(result, save_json)
        result["artifacts"]["result_json"] = path

    return result
