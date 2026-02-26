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
    defect_meta: Optional[Dict[str, torch.Tensor]] = None,
    save_path: str,
    title_prefix: str = "Diffusion Eval",
    show_shared_compare: bool = True,
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

    has_defects = isinstance(defect_meta, dict) and len(defect_meta) > 0
    ncols = 8 if show_shared_compare else 6
    if has_defects:
        ncols += 3
    fig, axes = plt.subplots(n, ncols, figsize=(4 * ncols, 4 * n))
    if n == 1:
        axes = axes[None, :]

    def _sym_abs_max(x: torch.Tensor, eps: float = 1.0e-6) -> float:
        return max(eps, float(x.detach().abs().max().item()))

    def _to_panel_numpy(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.detach().cpu()

    def _get_defect_sample(i: int, key: str) -> Optional[torch.Tensor]:
        if not has_defects:
            return None
        val = defect_meta.get(key)
        if not torch.is_tensor(val):
            return None
        if val.dim() >= 4 and val.shape[0] >= n:
            return val[i]
        return val

    for i in range(n):
        v_true_i = V_true[i]
        v_pred_i = V_pred[i]
        if v_true_i.dim() == 3 and v_true_i.shape[0] == 1:
            v_true_i = v_true_i[0]
        if v_pred_i.dim() == 3 and v_pred_i.shape[0] == 1:
            v_pred_i = v_pred_i[0]
        err = (v_pred_i - v_true_i).abs()
        gt_abs = _sym_abs_max(v_true_i)
        pred_abs = _sym_abs_max(v_pred_i)
        shared_abs = max(gt_abs, pred_abs)

        col = 0
        axes[i, col].imshow(
            v_true_i.detach().cpu().numpy(),
            cmap="inferno",
            vmin=-gt_abs,
            vmax=gt_abs,
            interpolation="nearest",
        )
        axes[i, col].set_title(
            f"{title_prefix}: GT auto\n[{v_true_i.min().item():.2f}, {v_true_i.max().item():.2f}]"
        )
        col += 1
        axes[i, col].imshow(
            v_pred_i.detach().cpu().numpy(),
            cmap="inferno",
            vmin=-pred_abs,
            vmax=pred_abs,
            interpolation="nearest",
        )
        axes[i, col].set_title(
            f"Pred auto\n[{v_pred_i.min().item():.2f}, {v_pred_i.max().item():.2f}]"
        )
        col += 1
        axes[i, col].imshow(err.detach().cpu().numpy(), cmap="magma", interpolation="nearest")
        axes[i, col].set_title(f"Abs Err {float(err.mean().item()):.4f}")
        col += 1

        if show_shared_compare:
            axes[i, col].imshow(
                v_true_i.detach().cpu().numpy(),
                cmap="inferno",
                vmin=-shared_abs,
                vmax=shared_abs,
                interpolation="nearest",
            )
            axes[i, col].set_title(f"GT shared ±{shared_abs:.2f}")
            col += 1
            axes[i, col].imshow(
                v_pred_i.detach().cpu().numpy(),
                cmap="inferno",
                vmin=-shared_abs,
                vmax=shared_abs,
                interpolation="nearest",
            )
            axes[i, col].set_title(f"Pred shared ±{shared_abs:.2f}")
            col += 1
        if sublattice_resolved:
            axes[i, col].imshow(g_obs_agg[i, k0].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS Agg E{k0}")
            col += 1
            axes[i, col].imshow(g_obs[i, k1, 0].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS A E{k1}")
            col += 1
            axes[i, col].imshow(g_obs[i, k1, 1].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS B E{k1}")
            col += 1
        else:
            axes[i, col].imshow(g_obs[i, k0].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS E{k0}")
            col += 1
            axes[i, col].imshow(g_obs[i, k1].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS E{k1}")
            col += 1
            axes[i, col].imshow(g_obs[i, k2].detach().cpu().numpy(), cmap="viridis")
            axes[i, col].set_title(f"LDOS E{k2}")
            col += 1

        if has_defects:
            vacancy_mask = _get_defect_sample(i, "vacancy_mask")
            vacancy_panel: Optional[torch.Tensor] = None
            if torch.is_tensor(vacancy_mask):
                if vacancy_mask.dim() == 3 and vacancy_mask.shape[0] == 2:
                    vacancy_panel = vacancy_mask.bool().any(dim=0).float()
                elif vacancy_mask.dim() == 2:
                    vacancy_panel = vacancy_mask.float()
            if vacancy_panel is None:
                vacancy_panel = torch.zeros_like(v_true_i, dtype=torch.float32)
            axes[i, col].imshow(_to_panel_numpy(vacancy_panel).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axes[i, col].set_title("vacancy_mask (A|B)")
            col += 1

            bond_mod = _get_defect_sample(i, "bond_mod")
            bond_panel: Optional[torch.Tensor] = None
            if torch.is_tensor(bond_mod):
                if bond_mod.dim() == 3 and bond_mod.shape[0] == 3:
                    bond_panel = bond_mod.abs().mean(dim=0)
                elif bond_mod.dim() == 2:
                    bond_panel = bond_mod.abs()
            if bond_panel is None:
                bond_panel = torch.zeros_like(v_true_i, dtype=torch.float32)
            axes[i, col].imshow(_to_panel_numpy(bond_panel).numpy(), cmap="cividis", interpolation="nearest")
            axes[i, col].set_title("bond_mod |.| mean")
            col += 1

            onsite_ab = _get_defect_sample(i, "onsite_ab_delta")
            onsite_panel: Optional[torch.Tensor] = None
            if torch.is_tensor(onsite_ab):
                if onsite_ab.dim() == 3 and onsite_ab.shape[0] == 2:
                    onsite_panel = onsite_ab[0] - onsite_ab[1]
                elif onsite_ab.dim() == 2:
                    onsite_panel = onsite_ab
            if onsite_panel is None:
                onsite_panel = torch.zeros_like(v_true_i, dtype=torch.float32)
            onsite_abs = _sym_abs_max(onsite_panel)
            axes[i, col].imshow(
                _to_panel_numpy(onsite_panel).numpy(),
                cmap="coolwarm",
                vmin=-onsite_abs,
                vmax=onsite_abs,
                interpolation="nearest",
            )
            axes[i, col].set_title("onsite_ab_delta (A-B)")
            col += 1

        for j in range(ncols):
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
