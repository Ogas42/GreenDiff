from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.obs_layout import aggregate_sublattice_ldos


def build_teacher_sampler_for_eval(
    config: Dict[str, Any],
    diffusion_model: torch.nn.Module,
    vae: torch.nn.Module,
    latent_green: torch.nn.Module,
) -> TeacherSampler:
    cond_enc = diffusion_model.condition_encoder if hasattr(diffusion_model, "condition_encoder") else None
    return TeacherSampler(
        config,
        diffusion_model=diffusion_model,
        vae=vae,
        condition_encoder=cond_enc,
        latent_green=latent_green,
    )


def sample_diffusion_predictions(
    *,
    sampler: TeacherSampler,
    g_obs: torch.Tensor,
    latent_unscale_factor: float | torch.Tensor | None = None,
) -> torch.Tensor:
    if latent_unscale_factor is not None:
        sampler.unscale_factor = latent_unscale_factor
    with torch.no_grad():
        return sampler.sample(g_obs)


def compute_diffusion_eval_metrics(V_pred: torch.Tensor, V_true: torch.Tensor) -> Dict[str, float]:
    pred = V_pred
    true = V_true
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if true.dim() == 3:
        true = true.unsqueeze(1)
    mse = float(F.mse_loss(pred, true).item())
    mae = float(F.l1_loss(pred, true).item())
    diff_norm = torch.norm((pred - true).reshape(pred.shape[0], -1), p=2, dim=1)
    true_norm = torch.norm(true.reshape(true.shape[0], -1), p=2, dim=1).clamp_min(1.0e-6)
    rel_l2 = float((diff_norm / true_norm).mean().item())

    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    true_fft = torch.fft.rfft2(true, dim=(-2, -1), norm="ortho")
    pred_psd = pred_fft.abs() ** 2
    true_psd = true_fft.abs() ** 2
    psd_error = float(F.l1_loss(torch.log(pred_psd + 1.0e-8), torch.log(true_psd + 1.0e-8)).item())
    return {"mse": mse, "mae": mae, "rel_l2": rel_l2, "psd_error": psd_error}


def render_diffusion_comparison_grid(
    *,
    V_true: torch.Tensor,
    V_pred: torch.Tensor,
    g_obs: torch.Tensor,
    save_path: str,
    title_prefix: str = "Diffusion Eval",
) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    n = min(V_true.shape[0], V_pred.shape[0], g_obs.shape[0])
    n = int(max(1, n))
    k0 = 0
    k1 = int(g_obs.shape[1] // 2)
    k2 = int(g_obs.shape[1] - 1)
    sublattice_resolved = g_obs.dim() == 5
    g_obs_agg = aggregate_sublattice_ldos(g_obs) if sublattice_resolved else g_obs

    fig, axes = plt.subplots(n, 6, figsize=(22, 4 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        v_true_i = V_true[i]
        v_pred_i = V_pred[i]
        if v_true_i.dim() == 3 and v_true_i.shape[0] == 1:
            v_true_i = v_true_i[0]
        if v_pred_i.dim() == 3 and v_pred_i.shape[0] == 1:
            v_pred_i = v_pred_i[0]
        err = (v_pred_i - v_true_i).abs()

        axes[i, 0].imshow(v_true_i.detach().cpu().numpy(), cmap="inferno")
        axes[i, 0].set_title(f"{title_prefix}: GT")
        axes[i, 1].imshow(v_pred_i.detach().cpu().numpy(), cmap="inferno")
        axes[i, 1].set_title("Pred")
        axes[i, 2].imshow(err.detach().cpu().numpy(), cmap="magma")
        axes[i, 2].set_title(f"Abs Err {float(err.mean().item()):.4f}")
        if sublattice_resolved:
            axes[i, 3].imshow(g_obs_agg[i, k0].detach().cpu().numpy(), cmap="viridis")
            axes[i, 3].set_title(f"LDOS Agg E{k0}")
            axes[i, 4].imshow(g_obs[i, k1, 0].detach().cpu().numpy(), cmap="viridis")
            axes[i, 4].set_title(f"LDOS A E{k1}")
            axes[i, 5].imshow(g_obs[i, k1, 1].detach().cpu().numpy(), cmap="viridis")
            axes[i, 5].set_title(f"LDOS B E{k1}")
        else:
            axes[i, 3].imshow(g_obs[i, k0].detach().cpu().numpy(), cmap="viridis")
            axes[i, 3].set_title(f"LDOS E{k0}")
            axes[i, 4].imshow(g_obs[i, k1].detach().cpu().numpy(), cmap="viridis")
            axes[i, 4].set_title(f"LDOS E{k1}")
            axes[i, 5].imshow(g_obs[i, k2].detach().cpu().numpy(), cmap="viridis")
            axes[i, 5].set_title(f"LDOS E{k2}")
        for j in range(6):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)
    return save_path


def summarize_metric_lists(metric_lists: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for k, vals in metric_lists.items():
        if not vals:
            out[f"{k}_mean"] = None
            out[f"{k}_std"] = None
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        out[f"{k}_mean"] = float(t.mean().item())
        out[f"{k}_std"] = float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0
    for k in ("mse", "mae", "rel_l2", "psd_error"):
        out[k] = out.get(f"{k}_mean")
    return out
