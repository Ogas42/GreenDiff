from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Tuple

import torch

from gd.utils.ldos_transform import ldos_linear_from_obs, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import flatten_sub_for_energy_ops, g_obs_to_canonical_view, is_sublattice_resolved


def _green_physics_weights(model_cfg: Dict[str, Any]) -> Dict[str, float]:
    phy = model_cfg.get("physics_losses", {}) if isinstance(model_cfg.get("physics_losses", {}), dict) else {}
    return {
        "data_weight": float(phy.get("data_weight", 1.0)),
        "residual_weight": float(phy.get("residual_weight", model_cfg.get("residual_loss_weight", 0.0))),
        "sum_rule_weight": float(phy.get("sum_rule_weight", 0.0)),
        "nonneg_weight": float(phy.get("nonneg_weight", 0.0)),
    }


def build_green_scheduler(opt: torch.optim.Optimizer, train_cfg: Dict[str, Any]):
    sched_cfg = train_cfg.get("lr_schedule", {})
    if not sched_cfg.get("enabled"):
        return None
    max_steps = int(train_cfg["max_steps"])
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
    warmup_min_ratio = float(sched_cfg.get("warmup_min_ratio", min_lr_ratio))

    def lr_lambda(step_idx: int):
        if warmup_steps > 0 and step_idx < warmup_steps:
            ramp = step_idx / float(max(1, warmup_steps))
            return max(warmup_min_ratio, ramp)
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step_idx - warmup_steps) / float(max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def compute_rel_for_logging(
    cfg: Dict[str, Any],
    g_pred: torch.Tensor,
    g_obs: torch.Tensor,
    data_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    ldos_cfg = data_cfg.get("ldos_transform", {})
    log_cfg = ldos_cfg.get("log", {})
    log_enabled = bool(ldos_cfg.get("enabled", False) and log_cfg.get("enabled", False))
    model_cfg = cfg["latent_green"]["model"]
    loss_type = model_cfg.get("loss_type", "mse")
    huber_beta = float(model_cfg.get("huber_beta", 0.1))
    use_per_energy_affine = bool(model_cfg.get("per_energy_affine", False))
    align_cfg = model_cfg.get("energy_align", {})
    align_enabled = bool(align_cfg.get("enabled", False))
    align_max_shift = int(align_cfg.get("max_shift", 0))
    log_cosh_eps = float(model_cfg.get("log_cosh_eps", 1.0e-6))

    aux: Dict[str, Any] = {"log_enabled": log_enabled}
    sublattice_resolved = bool(is_sublattice_resolved(data_cfg))
    if log_enabled:
        eps = float(log_cfg.get("eps", 1.0e-6))
        if sublattice_resolved:
            g_pred_log = flatten_sub_for_energy_ops(
                g_obs_to_canonical_view(torch.log(g_pred.clamp_min(0) + eps), data_cfg)
            )
            g_obs_log = flatten_sub_for_energy_ops(g_obs)
        else:
            g_pred_log = torch.log(g_pred.clamp_min(0) + eps)
            g_obs_log = g_obs
        if use_per_energy_affine:
            g_pred_log = per_energy_affine(g_pred_log, g_obs_log)
        g_pred_log, _ = align_pred(
            g_pred_log,
            g_obs_log,
            enabled=align_enabled,
            max_shift=align_max_shift,
            loss_type=loss_type,
            huber_beta=huber_beta,
            log_cosh_eps=log_cosh_eps,
        )
        rel_l2 = torch.norm(g_pred_log - g_obs_log) / torch.norm(g_obs_log).clamp_min(1.0e-6)
        g_obs_lin = ldos_linear_from_obs(g_obs_log, data_cfg)
        aux.update(
            {
                "pred_min": g_pred.min().item(),
                "pred_max": g_pred.max().item(),
                "pred_min_log": g_pred_log.min().item(),
                "pred_max_log": g_pred_log.max().item(),
                "pred_mean_log": g_pred_log.mean().item(),
                "pred_std_log": g_pred_log.std().item(),
                "obs_min_log": g_obs_log.min().item(),
                "obs_max_log": g_obs_log.max().item(),
                "obs_mean_log": g_obs_log.mean().item(),
                "obs_std_log": g_obs_log.std().item(),
                "obs_lin_min": g_obs_lin.min().item(),
                "obs_lin_max": g_obs_lin.max().item(),
            }
        )
    else:
        if sublattice_resolved:
            g_pred_obs = flatten_sub_for_energy_ops(g_obs_to_canonical_view(ldos_obs_from_linear(g_pred, data_cfg), data_cfg))
            g_obs_obs = flatten_sub_for_energy_ops(g_obs)
            g_pred_lin_stats = flatten_sub_for_energy_ops(g_obs_to_canonical_view(g_pred, data_cfg))
            g_obs_lin = flatten_sub_for_energy_ops(g_obs_to_canonical_view(ldos_linear_from_obs(g_obs, data_cfg), data_cfg))
        else:
            g_pred_obs = ldos_obs_from_linear(g_pred, data_cfg)
            g_obs_obs = g_obs
            g_pred_lin_stats = g_pred
            g_obs_lin = ldos_linear_from_obs(g_obs, data_cfg)
        if use_per_energy_affine:
            g_pred_obs = per_energy_affine(g_pred_obs, g_obs_obs)
        g_pred_obs, _ = align_pred(
            g_pred_obs,
            g_obs_obs,
            enabled=align_enabled,
            max_shift=align_max_shift,
            loss_type=loss_type,
            huber_beta=huber_beta,
            log_cosh_eps=log_cosh_eps,
        )
        rel_l2 = torch.norm(g_pred_obs - g_obs_obs) / torch.norm(g_obs_obs).clamp_min(1.0e-6)
        aux.update(
            {
                "pred_min": g_pred_obs.min().item(),
                "pred_max": g_pred_obs.max().item(),
                "obs_min": g_obs_obs.min().item(),
                "obs_max": g_obs_obs.max().item(),
                "pred_lin_mean": g_pred_lin_stats.mean().item(),
                "pred_lin_std": g_pred_lin_stats.std().item(),
                "obs_lin_mean": g_obs_lin.mean().item(),
                "obs_lin_std": g_obs_lin.std().item(),
            }
        )
    return rel_l2, aux


def compute_green_loss(
    *,
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    model_core: torch.nn.Module,
    z: torch.Tensor,
    V: torch.Tensor,
    g_obs: torch.Tensor,
    physics_meta: Dict[str, Any] | None,
    defect_meta: Dict[str, Any] | None,
    noisy_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    step: int,
    device: torch.device,
    amp: Any,
):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type, enabled=bool(amp.use_amp), dtype=amp.amp_dtype):
        if noisy_cfg.get("enabled", False):
            t = torch.randint(0, noisy_cfg["T"], (z.shape[0],), device=device)
            z_t, _, _ = model_core.add_noise(z, t)
            clean_prob = float(noisy_cfg.get("clean_prob", 0.0))
            if clean_prob > 0.0:
                clean_mask = torch.rand(z.shape[0], device=device) < clean_prob
                if clean_mask.any():
                    t = t.clone()
                    t[clean_mask] = 0
                    z_t = z_t.clone()
                    z_t[clean_mask] = z[clean_mask]
            g_pred, psi_real, psi_imag, src = model(
                z_t, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True
            )
        else:
            t = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
            g_pred, psi_real, psi_imag, src = model(
                z, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True
            )
        residual_loss = model_core.residual_loss(psi_real, psi_imag, src, V, physics_meta=physics_meta, defect_meta=defect_meta)
        losses = model_core.loss(g_pred, g_obs, residual_loss, physics_meta=physics_meta)

    aux_warmup_steps = int(model_cfg.get("aux_warmup_steps", 0))
    aux_scale = min(1.0, step / float(aux_warmup_steps)) if aux_warmup_steps > 0 else 1.0
    fft_weight = model_cfg.get("fft_loss_weight", 0.0) if model_cfg.get("use_fft_loss", False) else 0.0
    stats_weight = model_cfg.get("stats_loss_weight", 0.0)
    psd_weight = model_cfg.get("psd_loss_weight", 0.0)
    linear_scale_weight = model_cfg.get("linear_scale_loss_weight", 0.0)
    ms_weight = model_cfg.get("multiscale_loss_weight", 0.0)
    peak_cfg = model_cfg.get("peak_control", {})
    if not isinstance(peak_cfg, dict):
        peak_cfg = {}
    peak_enabled = bool(peak_cfg.get("enabled", False))
    peak_log_w = float(peak_cfg.get("log_aux_weight", 0.0)) if peak_enabled else 0.0
    peak_topk_w = float(peak_cfg.get("topk_loss_weight", 0.0)) if peak_enabled else 0.0
    peak_ratio_w = float(peak_cfg.get("peak_ratio_penalty_weight", 0.0)) if peak_enabled else 0.0
    phy_w = _green_physics_weights(model_cfg)

    total_loss = phy_w["data_weight"] * losses["data_loss"]
    total_loss = total_loss + aux_scale * fft_weight * losses["fft_loss"]
    total_loss = total_loss + aux_scale * psd_weight * losses.get("psd_loss", torch.zeros_like(losses["data_loss"]))
    total_loss = total_loss + aux_scale * stats_weight * losses["stats_loss"]
    total_loss = total_loss + linear_scale_weight * losses["linear_scale_loss"]
    total_loss = total_loss + aux_scale * ms_weight * losses["ms_loss"]
    total_loss = total_loss + aux_scale * peak_log_w * losses.get("log_aux_loss", torch.zeros_like(losses["data_loss"]))
    total_loss = total_loss + aux_scale * peak_topk_w * losses.get("topk_peak_loss", torch.zeros_like(losses["data_loss"]))
    total_loss = total_loss + aux_scale * peak_ratio_w * losses.get("peak_ratio_penalty", torch.zeros_like(losses["data_loss"]))
    total_loss = total_loss + phy_w["residual_weight"] * losses["residual_loss"]
    total_loss = total_loss + phy_w["sum_rule_weight"] * losses.get("sum_rule_loss", torch.zeros_like(losses["data_loss"]))
    total_loss = total_loss + phy_w["nonneg_weight"] * losses.get("nonneg_loss", torch.zeros_like(losses["data_loss"]))
    losses["loss"] = total_loss

    rel_l2, aux_log = compute_rel_for_logging(cfg, g_pred.detach(), g_obs.detach(), data_cfg)
    residual_aux = getattr(model_core, "get_last_residual_aux", lambda: {})() or {}
    if "residual_active_frac" in residual_aux:
        aux_log["residual_active_frac"] = float(residual_aux["residual_active_frac"])
    if isinstance(physics_meta, dict) and "hopping" in physics_meta:
        hop = physics_meta["hopping"].detach().float()
        aux_log["hopping_mean"] = float(hop.mean().item())
        aux_log["hopping_std"] = float(hop.std().item()) if hop.numel() > 1 else 0.0
    return losses, rel_l2, aux_log


def green_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    losses: Dict[str, torch.Tensor],
    grad_clip: float,
    amp: Any,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if amp.use_scaler and amp.scaler is not None:
        amp.scaler.scale(losses["loss"]).backward()
        if grad_clip > 0:
            amp.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()
    else:
        losses["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()


def log_green_train_status(
    *,
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    g_obs: torch.Tensor,
    losses: Dict[str, torch.Tensor],
    rel_l2: torch.Tensor,
    aux_log: Dict[str, Any],
    loss_hist: deque,
    data_hist: deque,
    rel_hist: deque,
    it_s: float,
    samples_s: float,
    pbar: Any,
) -> None:
    train_cfg = cfg["latent_green"]["training"]
    loss_hist.append(losses["loss"].item())
    data_hist.append(losses["data_loss"].item())
    rel_hist.append(rel_l2.item())
    loss_avg = sum(loss_hist) / float(len(loss_hist))
    data_avg = sum(data_hist) / float(len(data_hist))
    rel_avg = sum(rel_hist) / float(len(rel_hist))
    denom = max(1.0, float(g_obs[0].numel()))
    loss_norm = losses["loss"].item() / denom
    data_norm = losses["data_loss"].item() / denom
    lr_val = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
    grad_norm = math.sqrt(grad_norm)
    postfix_parts = [
        f"dom {str(cfg['latent_green']['model'].get('data_loss_domain', 'obs_legacy'))[:4]}",
        f"loss {losses['loss'].item():.4f}",
        f"data {losses['data_loss'].item():.4f}",
        f"fft {losses['fft_loss'].item():.4f}",
        f"rel {rel_l2.item():.4f}",
        f"loss_n {loss_norm:.6f}",
        f"data_n {data_norm:.6f}",
        f"loss_avg {loss_avg:.4f}",
        f"data_avg {data_avg:.4f}",
        f"rel_avg {rel_avg:.4f}",
        f"lr {lr_val:.2e}",
        f"gn {grad_norm:.3f}",
        f"it/s {it_s:.2f}",
        f"s/s {samples_s:.0f}",
    ]
    for key, short in [("stats_loss", "stats"), ("psd_loss", "psd"), ("ms_loss", "ms"), ("residual_loss", "res"), ("sum_rule_loss", "sum")]:
        if key in losses:
            postfix_parts.append(f"{short} {losses[key].item():.4f}")
    if "residual_active_frac" in aux_log:
        postfix_parts.append(f"ract {aux_log['residual_active_frac']:.3f}")
    if "hopping_mean" in aux_log:
        postfix_parts.append(f"t {aux_log['hopping_mean']:.3f}")
        postfix_parts.append(f"tstd {aux_log.get('hopping_std', 0.0):.3f}")
    summary = f"Step {step}: " + " | ".join(postfix_parts)

    log_every = int(train_cfg["log_every"])
    if step % (log_every * 5) == 0:
        if aux_log.get("log_enabled", False):
            msg = (
                f"Step {step}: "
                f"Pred Lin [{aux_log.get('pred_min', 0):.3f}, {aux_log.get('pred_max', 0):.3f}] "
                f"Log [{aux_log.get('pred_min_log', 0):.3f}, {aux_log.get('pred_max_log', 0):.3f}] | "
                f"Obs Lin [{aux_log.get('obs_lin_min', 0):.3f}, {aux_log.get('obs_lin_max', 0):.3f}] "
                f"Log [{aux_log.get('obs_min_log', 0):.3f}, {aux_log.get('obs_max_log', 0):.3f}]"
            )
            stats_msg = (
                f"Step {step}: "
                f"Pred Log mean/std [{aux_log.get('pred_mean_log', 0):.3f}, {aux_log.get('pred_std_log', 0):.3f}] | "
                f"Obs Log mean/std [{aux_log.get('obs_mean_log', 0):.3f}, {aux_log.get('obs_std_log', 0):.3f}]"
            )
        else:
            msg = (
                f"Step {step}: Pred Obs [{aux_log.get('pred_min', 0):.3f}, {aux_log.get('pred_max', 0):.3f}] | "
                f"Obs Obs [{aux_log.get('obs_min', 0):.3f}, {aux_log.get('obs_max', 0):.3f}]"
            )
            stats_msg = (
                f"Step {step}: Pred Lin mean/std [{aux_log.get('pred_lin_mean', 0):.3f}, {aux_log.get('pred_lin_std', 0):.3f}] | "
                f"Obs Lin mean/std [{aux_log.get('obs_lin_mean', 0):.3f}, {aux_log.get('obs_lin_std', 0):.3f}]"
            )
        if pbar is not None:
            pbar.write(summary)
            pbar.write(msg)
            pbar.write(stats_msg)
        else:
            print(summary)
            print(msg)
            print(stats_msg)
    elif pbar is not None:
        pbar.write(summary)
    else:
        print(summary)

    if pbar is not None and hasattr(pbar, "set_postfix_str"):
        pbar.set_postfix_str("")

    check_cfg = train_cfg.get("surrogate_check", {})
    if check_cfg.get("enabled", False):
        rel_max = float(check_cfg.get("rel_l2_max", 0.3))
        warmup_steps = int(check_cfg.get("warmup_steps", 0))
        if step >= warmup_steps and rel_l2.item() > rel_max:
            print(f"Warning: surrogate mismatch rel_l2={rel_l2.item():.4f} > {rel_max:.4f}")
