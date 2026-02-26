from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from gd.core.logging.results import append_train_metric_jsonl
from gd.utils.ldos_transform import ldos_linear_from_obs, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import flatten_sub_for_energy_ops, g_obs_to_canonical_view, is_sublattice_resolved


def build_diffusion_scheduler(opt: torch.optim.Optimizer, train_cfg: Dict[str, Any]):
    sched_cfg = train_cfg.get("lr_schedule", {})
    if not sched_cfg.get("enabled", False):
        return None

    schedule_type = str(sched_cfg.get("type", "cosine"))
    max_steps = int(train_cfg["max_steps"])
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
    warmup_min_ratio = float(sched_cfg.get("warmup_min_ratio", min_lr_ratio))

    def lr_lambda(step_idx: int):
        if warmup_steps > 0 and step_idx < warmup_steps:
            ramp = step_idx / float(max(1, warmup_steps))
            return max(warmup_min_ratio, ramp)
        if schedule_type != "cosine" or max_steps <= warmup_steps:
            return 1.0
        progress = (step_idx - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def prepare_latent_batch(
    *,
    vae: torch.nn.Module,
    V: torch.Tensor,
    train_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, Any]]:
    latent_scale_cfg = train_cfg.get("latent_scale", {})
    latent_scale_mode = latent_scale_cfg.get("mode", "none")
    latent_target_std = float(latent_scale_cfg.get("target_std", 1.0))
    latent_scale = float(latent_scale_cfg.get("scale", 1.0))
    latent_eps = float(latent_scale_cfg.get("eps", 1.0e-6))

    z, mu, _ = vae.encode(V)
    if mu is not None:
        z = mu
    unscale_factor: float | torch.Tensor = 1.0
    if latent_scale_mode == "auto":
        z_std = z.std(dim=(1, 2, 3), keepdim=True).clamp_min(latent_eps)
        scale = latent_target_std / z_std
        z = z * scale
        unscale_factor = 1.0 / scale
    elif latent_scale_mode == "fixed":
        z = z * latent_scale
        unscale_factor = 1.0 / max(1.0e-6, latent_scale)
    return z, {"latent_scale_mode": latent_scale_mode, "unscale_factor": unscale_factor}


def sample_diffusion_training_target(
    *,
    model_core: torch.nn.Module,
    z: torch.Tensor,
    prediction_type: str,
) -> Dict[str, torch.Tensor]:
    device = z.device
    t = torch.randint(0, int(model_core.T), (z.shape[0],), device=device)
    noise = torch.randn_like(z)
    alpha_t_1d, sigma_t_1d = model_core.get_alpha_sigma(t)
    alpha_t = alpha_t_1d.view(-1, 1, 1, 1)
    sigma_t = sigma_t_1d.view(-1, 1, 1, 1)
    z_t = alpha_t * z + sigma_t * noise
    if prediction_type == "v":
        target = alpha_t * noise - sigma_t * z
    elif prediction_type == "x0":
        target = z
    else:
        target = noise
    snr = (alpha_t_1d**2) / (sigma_t_1d**2).clamp_min(1.0e-8)
    return {
        "t": t,
        "noise": noise,
        "alpha_t": alpha_t,
        "sigma_t": sigma_t,
        "alpha_t_1d": alpha_t_1d,
        "sigma_t_1d": sigma_t_1d,
        "z_t": z_t,
        "target": target,
        "snr": snr,
    }


def compute_diffusion_base_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    snr: torch.Tensor,
    train_cfg: Dict[str, Any],
    prediction_type: str,
    step: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    min_snr_cfg = train_cfg.get("min_snr", {})
    use_min_snr = bool(min_snr_cfg.get("enabled", False))
    min_snr_gamma = float(min_snr_cfg.get("gamma", 5.0))
    lr_sched_cfg = train_cfg.get("lr_schedule", {})
    warmup_steps = int(lr_sched_cfg.get("warmup_steps", 0))

    if use_min_snr:
        gamma_eff = min_snr_gamma
        if step < warmup_steps:
            gamma_eff = min_snr_gamma + (5.0 - min_snr_gamma) * (
                1.0 - float(step + 1) / float(max(1, warmup_steps))
            )
        weight_denom = snr + 1.0 if prediction_type == "v" else snr
        weight = torch.minimum(snr, torch.full_like(snr, gamma_eff)) / weight_denom.clamp_min(1.0e-8)
        per_sample = (pred - target) ** 2
        per_sample = per_sample.mean(dim=(1, 2, 3))
        base_loss = (weight * per_sample).mean()
        return base_loss, {"weight_mean": float(weight.mean().item()), "use_min_snr": True, "snr_weight": weight.detach()}
    return F.mse_loss(pred, target), {"weight_mean": 1.0, "use_min_snr": False, "snr_weight": None}


def compute_x0_pred(
    *,
    prediction_type: str,
    z_t: torch.Tensor,
    pred: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "v":
        return alpha_t * z_t - sigma_t * pred
    if prediction_type == "x0":
        return pred
    return (z_t - sigma_t * pred) / alpha_t.clamp_min(1.0e-6)


def _psd_loss_per_sample(pred: torch.Tensor, obs: torch.Tensor, psd_eps: float) -> torch.Tensor:
    if pred.dim() == 5:
        pred = flatten_sub_for_energy_ops(pred)
    if obs.dim() == 5:
        obs = flatten_sub_for_energy_ops(obs)
    pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    obs_f = torch.fft.rfft2(obs, dim=(-2, -1), norm="ortho")
    pred_p = pred_f.real**2 + pred_f.imag**2
    obs_p = obs_f.real**2 + obs_f.imag**2
    pred_l = torch.log(pred_p + psd_eps)
    obs_l = torch.log(obs_p + psd_eps)
    return F.l1_loss(pred_l, obs_l, reduction="none").mean(dim=(1, 2, 3))


def _normalize_energy_weights(w: torch.Tensor, *, eps: float, power: float) -> torch.Tensor:
    w = w.clamp_min(eps)
    if power != 1.0:
        w = w**power
    scale = w.numel() / w.sum().clamp_min(eps)
    return w * scale


def _energy_weights_from_obs(
    obs: torch.Tensor,
    *,
    mode: str,
    eps: float,
    power: float,
) -> Optional[torch.Tensor]:
    if obs.dim() == 5:
        obs = flatten_sub_for_energy_ops(obs)
    if mode == "snr":
        mean = obs.abs().mean(dim=(0, 2, 3))
        std = obs.std(dim=(0, 2, 3)).clamp_min(eps)
        return _normalize_energy_weights(mean / std, eps=eps, power=power)
    if mode == "sensitivity":
        return _normalize_energy_weights(obs.std(dim=(0, 2, 3)), eps=eps, power=power)
    return None


def _rms_normalize_pair(
    pred: torch.Tensor,
    obs: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(1, obs.dim()))
    scale = torch.sqrt((obs**2).mean(dim=reduce_dims, keepdim=True) + eps).clamp_min(eps)
    return pred / scale, obs / scale, scale


def compute_physics_terms(
    *,
    x0_pred: torch.Tensor,
    latent_green: torch.nn.Module,
    g_obs: torch.Tensor,
    alpha_t: torch.Tensor,
    train_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    compute_consistency: bool,
) -> Dict[str, Any]:
    device = x0_pred.device
    t_zeros = torch.zeros((x0_pred.shape[0],), dtype=torch.long, device=device)

    phys_loss_type = str(train_cfg.get("phys_loss_type", "mse"))
    huber_beta = float(train_cfg.get("huber_beta", 0.1))
    use_per_energy_affine = bool(train_cfg.get("per_energy_affine", False))
    align_cfg = train_cfg.get("energy_align", {})
    align_enabled = bool(align_cfg.get("enabled", False))
    align_max_shift = int(align_cfg.get("max_shift", 0))
    log_cosh_eps = float(train_cfg.get("log_cosh_eps", 1.0e-6))
    psd_loss_weight = float(train_cfg.get("psd_loss_weight", 0.0))
    psd_eps = float(train_cfg.get("psd_eps", 1.0e-8))
    consistency_loss_weight = float(train_cfg.get("consistency_loss_weight", 0.0) or 0.0)
    topk_cfg = train_cfg.get("topk_phys", {})
    topk_enabled = bool(topk_cfg.get("enabled", False))
    topk_k = int(topk_cfg.get("k", 0) or 0)

    energy_weights_cfg = train_cfg.get("energy_weights", [])
    energy_weights = list(energy_weights_cfg) if isinstance(energy_weights_cfg, (list, tuple)) else []
    energy_weight_mode = str(train_cfg.get("energy_weight_mode", "uniform"))
    energy_weight_eps = float(train_cfg.get("energy_weight_eps", 1.0e-6))
    energy_weight_power = float(train_cfg.get("energy_weight_power", 1.0))

    phys_sup_cfg = train_cfg.get("phys_supervision", {})
    if not isinstance(phys_sup_cfg, dict):
        phys_sup_cfg = {}
    phys_sup_enabled = bool(phys_sup_cfg.get("enabled", True))
    phys_domain = str(phys_sup_cfg.get("domain", "linear_normalized" if phys_sup_enabled else "obs_legacy")).lower()
    normalize_rms = bool(phys_sup_cfg.get("normalize_per_sample_rms", True))
    consistency_on_normalized = bool(phys_sup_cfg.get("consistency_on_normalized_linear", True))

    sublattice_resolved = bool(is_sublattice_resolved(data_cfg))
    g_pred_lin_model = latent_green(x0_pred, t_zeros)

    if phys_domain == "linear_normalized":
        pred_eval = g_obs_to_canonical_view(g_pred_lin_model, data_cfg) if sublattice_resolved else g_pred_lin_model
        obs_lin = ldos_linear_from_obs(g_obs, data_cfg)
        obs_eval = obs_lin if sublattice_resolved else obs_lin
        if not sublattice_resolved:
            # keep shapes 4D for downstream loss helpers and energy weighting
            pred_eval = pred_eval
            obs_eval = obs_eval
        if normalize_rms:
            pred_eval, obs_eval, _scale = _rms_normalize_pair(pred_eval, obs_eval, eps=1.0e-6)
        if use_per_energy_affine:
            pred_eval = per_energy_affine(pred_eval, obs_eval)
        pred_eval, per_energy_loss = align_pred(
            pred_eval,
            obs_eval,
            enabled=align_enabled,
            max_shift=align_max_shift,
            loss_type=phys_loss_type,
            huber_beta=huber_beta,
            log_cosh_eps=log_cosh_eps,
        )
        obs_for_energy_weight = obs_eval
        pred_for_psd = pred_eval
        obs_for_psd = obs_eval
        compute_consistency_here = compute_consistency and consistency_on_normalized
        raw_phys_loss_obs_domain = None
    else:
        g_pred_phys_for_loss = ldos_obs_from_linear(g_pred_lin_model, data_cfg)
        if sublattice_resolved:
            g_pred_phys_for_loss = g_obs_to_canonical_view(g_pred_phys_for_loss, data_cfg)
        pred_eval = g_pred_phys_for_loss
        obs_eval = g_obs
        if use_per_energy_affine:
            pred_eval = per_energy_affine(pred_eval, obs_eval)
        pred_eval, per_energy_loss = align_pred(
            pred_eval,
            obs_eval,
            enabled=align_enabled,
            max_shift=align_max_shift,
            loss_type=phys_loss_type,
            huber_beta=huber_beta,
            log_cosh_eps=log_cosh_eps,
        )
        obs_for_energy_weight = obs_eval
        pred_for_psd = pred_eval
        obs_for_psd = obs_eval
        compute_consistency_here = compute_consistency
        raw_phys_loss_obs_domain = None

    if len(energy_weights) == g_obs.shape[1]:
        w = torch.tensor(energy_weights, device=device, dtype=per_energy_loss.dtype)
        w = _normalize_energy_weights(w, eps=energy_weight_eps, power=energy_weight_power).view(1, -1)
        per_energy_loss = per_energy_loss * w
    else:
        w_dyn = _energy_weights_from_obs(
            obs_for_energy_weight, mode=energy_weight_mode, eps=energy_weight_eps, power=energy_weight_power
        )
        if w_dyn is not None:
            per_energy_loss = per_energy_loss * w_dyn.view(1, -1)

    if topk_enabled and topk_k > 0:
        k_val = min(topk_k, per_energy_loss.shape[1])
        phys_loss_per_sample = torch.topk(per_energy_loss, k_val, dim=1).values.mean(dim=1)
    else:
        phys_loss_per_sample = per_energy_loss.mean(dim=1)

    phys_sample_weight = alpha_t.view(-1) ** 2
    phys_loss = (phys_loss_per_sample * phys_sample_weight).mean()
    psd_loss = None
    if psd_loss_weight > 0:
        psd_per_sample = _psd_loss_per_sample(pred_for_psd, obs_for_psd, psd_eps=psd_eps)
        psd_loss = (psd_per_sample * phys_sample_weight).mean()

    consistency_loss = None
    raw_consistency_loss = None
    if compute_consistency_here:
        pred_diff = pred_eval[:, 1:] - pred_eval[:, :-1]
        obs_diff = obs_eval[:, 1:] - obs_eval[:, :-1]
        const_map = F.mse_loss(pred_diff, obs_diff, reduction="none")
        const_per_sample = const_map.mean(dim=tuple(range(1, const_map.dim())))
        consistency_loss = (const_per_sample * phys_sample_weight).mean()
        raw_consistency_loss = float(const_per_sample.mean().item())

    phys_warmup_cfg = train_cfg.get("phys_warmup", {})
    phys_loss_weight = float(train_cfg.get("phys_loss_weight", 0.0))
    phys_coeff = phys_loss_weight
    if bool(phys_warmup_cfg.get("enabled", False)) and int(phys_warmup_cfg.get("warmup_steps", 0)) > 0:
        # step-dependent ramp is applied in caller since step is not available here
        pass

    return {
        "phys_loss": phys_loss,
        "psd_loss": psd_loss,
        "consistency_loss": consistency_loss,
        "raw_phys_loss": float(phys_loss_per_sample.mean().item()),
        "raw_consistency_loss": raw_consistency_loss,
        "raw_phys_loss_obs_domain": raw_phys_loss_obs_domain,
        "pred_for_consistency": pred_eval,
        "phys_eval_domain": phys_domain,
    }


def compute_total_diffusion_loss(
    *,
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    model_core: torch.nn.Module,
    latent_green: torch.nn.Module,
    z: torch.Tensor,
    g_obs: torch.Tensor,
    train_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    step: int,
    amp: Any,
    sample: Dict[str, torch.Tensor],
    phys_gate_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prediction_type = str(train_cfg.get("prediction_type", "eps"))
    x0_loss_weight = float(train_cfg.get("x0_loss_weight", 0.0))
    phys_loss_weight_cfg = float(train_cfg.get("phys_loss_weight", 0.0))
    phys_warmup_cfg = train_cfg.get("phys_warmup", {})
    psd_loss_weight = float(train_cfg.get("psd_loss_weight", 0.0))
    consistency_loss_weight_cfg = float(train_cfg.get("consistency_loss_weight", 0.0) or 0.0)
    phys_sup_cfg = train_cfg.get("phys_supervision", {})
    if not isinstance(phys_sup_cfg, dict):
        phys_sup_cfg = {}
    monitor_when_disabled = bool(phys_sup_cfg.get("monitor_when_disabled", True))
    log_every = int(train_cfg.get("log_every", 1) or 1)
    should_monitor_phys = bool(monitor_when_disabled and (step % max(1, log_every) == 0))

    gate_enabled = bool((phys_gate_state or {}).get("enabled", False))
    gate_passed = bool((phys_gate_state or {}).get("passed", True))
    if gate_enabled and not gate_passed:
        phys_loss_weight = 0.0
        consistency_loss_weight = 0.0
    else:
        phys_loss_weight = phys_loss_weight_cfg
        consistency_loss_weight = consistency_loss_weight_cfg
    phys_gate_reason = (phys_gate_state or {}).get("reason")

    device_type = "cuda" if z.device.type == "cuda" else "cpu"
    with torch.amp.autocast(device_type, enabled=bool(amp.use_amp), dtype=amp.amp_dtype):
        pred = model(sample["z_t"], sample["t"], g_obs)
        pred = pred.to(torch.float32)

        base_loss, base_aux = compute_diffusion_base_loss(
            pred=pred,
            target=sample["target"],
            snr=sample["snr"],
            train_cfg=train_cfg,
            prediction_type=prediction_type,
            step=step,
        )
        total_loss = base_loss

        x0_pred = None
        x0_loss = None
        raw_x0_loss = None
        need_phys_terms = phys_loss_weight > 0 or should_monitor_phys
        if x0_loss_weight > 0 or need_phys_terms:
            x0_pred = compute_x0_pred(
                prediction_type=prediction_type,
                z_t=sample["z_t"],
                pred=pred,
                alpha_t=sample["alpha_t"],
                sigma_t=sample["sigma_t"],
            )

        if x0_pred is not None:
            x0_loss_per_sample = F.mse_loss(x0_pred, z, reduction="none").mean(dim=(1, 2, 3))
            x0_sample_weight = sample["alpha_t"].view(-1) ** 2
            x0_loss = (x0_loss_per_sample * x0_sample_weight).mean()
            raw_x0_loss = float(F.mse_loss(x0_pred, z).item())
            if x0_loss_weight > 0:
                total_loss = total_loss + x0_loss_weight * x0_loss

        phys_loss = None
        psd_loss = None
        consistency_loss = None
        raw_phys_loss = None
        raw_consistency_loss = None
        raw_phys_loss_obs_domain = None
        phys_eval_domain = None
        phys_coeff = 0.0
        phys_loss_weight_eff = phys_loss_weight
        consistency_loss_weight_eff = consistency_loss_weight
        if x0_pred is not None and (phys_loss_weight > 0 or should_monitor_phys):
            phys_terms = compute_physics_terms(
                x0_pred=x0_pred,
                latent_green=latent_green,
                g_obs=g_obs,
                alpha_t=sample["alpha_t"],
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                compute_consistency=(consistency_loss_weight > 0 or should_monitor_phys),
            )
            phys_loss = phys_terms["phys_loss"]
            psd_loss = phys_terms["psd_loss"]
            consistency_loss = phys_terms["consistency_loss"]
            raw_phys_loss = phys_terms["raw_phys_loss"]
            raw_consistency_loss = phys_terms["raw_consistency_loss"]
            raw_phys_loss_obs_domain = phys_terms.get("raw_phys_loss_obs_domain")
            phys_eval_domain = phys_terms.get("phys_eval_domain")

            if phys_loss_weight > 0:
                total_phys = phys_loss
                if psd_loss is not None and psd_loss_weight > 0:
                    total_phys = total_phys + psd_loss_weight * psd_loss

                phys_coeff = phys_loss_weight
                if bool(phys_warmup_cfg.get("enabled", False)) and int(phys_warmup_cfg.get("warmup_steps", 0)) > 0:
                    warmup_steps = int(phys_warmup_cfg.get("warmup_steps", 0))
                    start_ratio = float(phys_warmup_cfg.get("start_ratio", 1.0))
                    end_ratio = float(phys_warmup_cfg.get("end_ratio", 1.0))
                    ramp = min(float(step + 1) / float(max(1, warmup_steps)), 1.0)
                    phys_coeff = phys_loss_weight * (start_ratio + (end_ratio - start_ratio) * ramp)

                total_loss = total_loss + phys_coeff * total_phys
                if consistency_loss is not None and consistency_loss_weight > 0:
                    total_loss = total_loss + consistency_loss_weight * consistency_loss

    return {
        "loss": total_loss,
        "base_loss": base_loss,
        "x0_loss": x0_loss,
        "phys_loss": phys_loss,
        "psd_loss": psd_loss,
        "consistency_loss": consistency_loss,
        "pred": pred,
        "x0_pred": x0_pred,
        "snr": sample["snr"],
        "weight_mean": base_aux["weight_mean"],
        "raw_x0_loss": raw_x0_loss,
        "raw_phys_loss": raw_phys_loss,
        "raw_consistency_loss": raw_consistency_loss,
        "raw_phys_loss_norm": raw_phys_loss,
        "raw_consistency_loss_norm": raw_consistency_loss,
        "raw_phys_loss_obs_domain": raw_phys_loss_obs_domain,
        "phys_eval_domain": phys_eval_domain,
        "phys_coeff": phys_coeff,
        "phys_loss_weight_eff": phys_loss_weight_eff,
        "consistency_loss_weight_eff": consistency_loss_weight_eff,
        "phys_gate_enabled": gate_enabled,
        "phys_gate_passed": gate_passed,
        "phys_gate_reason": phys_gate_reason,
    }


def diffusion_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss: torch.Tensor,
    grad_clip: float,
    amp: Any,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    if amp.use_scaler and amp.scaler is not None:
        amp.scaler.scale(loss).backward()
        if grad_clip > 0:
            amp.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()
    else:
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()


def update_ema_model(*, ema_model: torch.nn.Module | None, model_core: torch.nn.Module, ema_decay: float | None) -> None:
    if ema_model is None or ema_decay is None:
        return
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model_core.parameters()):
            ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
        for ema_b, b in zip(ema_model.buffers(), model_core.buffers()):
            ema_b.copy_(b)


def log_diffusion_train_status(
    *,
    step: int,
    train_cfg: Dict[str, Any],
    opt: torch.optim.Optimizer,
    z: torch.Tensor,
    loss_pack: Dict[str, Any],
    histories: Dict[str, deque],
    pbar: Any,
    start_or_last_log_time: float,
) -> float:
    now = time.perf_counter()
    elapsed = max(1.0e-6, now - start_or_last_log_time)
    log_every = int(train_cfg.get("log_every", 1))
    it_s = float(log_every) / elapsed if step > 0 else 0.0
    if step == 0:
        it_s = 0.0
    samples_s = it_s * float(z.shape[0])

    histories["loss"].append(float(loss_pack["loss"].detach().item()))
    histories["base"].append(float(loss_pack["base_loss"].detach().item()))
    histories["x0"].append(loss_pack["raw_x0_loss"])
    histories["phys"].append(loss_pack.get("raw_phys_loss_norm", loss_pack.get("raw_phys_loss")))
    histories["cons"].append(loss_pack.get("raw_consistency_loss_norm", loss_pack.get("raw_consistency_loss")))
    histories["phys_coeff"].append(loss_pack["phys_coeff"])

    def _avg(name: str):
        vals = [v for v in histories[name] if v is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else 0.0
    loss_val = float(loss_pack["loss"].detach().item())
    base_val = float(loss_pack["base_loss"].detach().item())
    x0_raw = loss_pack["raw_x0_loss"]
    phys_raw = loss_pack.get("raw_phys_loss_norm", loss_pack.get("raw_phys_loss"))
    cons_raw = loss_pack.get("raw_consistency_loss_norm", loss_pack.get("raw_consistency_loss"))
    snr = loss_pack["snr"]
    gate_tag = ""
    if loss_pack.get("phys_gate_enabled", False):
        gate_tag = " gate=on" if bool(loss_pack.get("phys_gate_passed", False)) else " gate=off"
    summary = (
        f"[diff] step={step} loss={loss_val:.6f} base={base_val:.6f} "
        f"x0={x0_raw if x0_raw is not None else 'na'} phys={phys_raw if phys_raw is not None else 'na'} "
        f"cons={cons_raw if cons_raw is not None else 'na'} lr={lr:.2e} "
        f"snr={float(snr.mean().item()):.2f}/{float(snr.min().item()):.2f}/{float(snr.max().item()):.2f} "
        f"w={float(loss_pack.get('weight_mean', 1.0)):.3f} it/s={it_s:.2f} s/s={samples_s:.0f}{gate_tag}"
    )
    avg_msg = (
        f"[diff-avg] loss={_avg('loss')} base={_avg('base')} x0={_avg('x0')} "
        f"phys={_avg('phys')} cons={_avg('cons')} phys_w={_avg('phys_coeff')}"
    )
    if pbar is not None:
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"loss": f"{loss_val:.6f}", "lr": f"{lr:.2e}", "snr": f"{float(snr.mean().item()):.2f}"})
        pbar.write(summary)
        pbar.write(avg_msg)
    else:
        print(summary)
        print(avg_msg)
    return now


def append_diffusion_train_metric_jsonl(
    *,
    path: str,
    step: int,
    opt: torch.optim.Optimizer,
    loss_pack: Dict[str, Any],
) -> str:
    rec = {
        "task": "diffusion_train",
        "stage": "diffusion",
        "step": int(step),
        "loss": float(loss_pack["loss"].detach().item()),
        "base_loss": float(loss_pack["base_loss"].detach().item()),
        "x0_loss": float(loss_pack["x0_loss"].detach().item()) if loss_pack.get("x0_loss") is not None else None,
        "phys_loss": float(loss_pack["phys_loss"].detach().item()) if loss_pack.get("phys_loss") is not None else None,
        "psd_loss": float(loss_pack["psd_loss"].detach().item()) if loss_pack.get("psd_loss") is not None else None,
        "consistency_loss": float(loss_pack["consistency_loss"].detach().item())
        if loss_pack.get("consistency_loss") is not None
        else None,
        "raw_x0_loss": loss_pack.get("raw_x0_loss"),
        "raw_phys_loss": loss_pack.get("raw_phys_loss"),
        "raw_consistency_loss": loss_pack.get("raw_consistency_loss"),
        "raw_phys_loss_norm": loss_pack.get("raw_phys_loss_norm", loss_pack.get("raw_phys_loss")),
        "raw_consistency_loss_norm": loss_pack.get("raw_consistency_loss_norm", loss_pack.get("raw_consistency_loss")),
        "raw_phys_loss_obs_domain": loss_pack.get("raw_phys_loss_obs_domain"),
        "phys_eval_domain": loss_pack.get("phys_eval_domain"),
        "snr_mean": float(loss_pack["snr"].mean().item()),
        "weight_mean": float(loss_pack.get("weight_mean", 1.0)),
        "phys_coeff": loss_pack.get("phys_coeff"),
        "phys_loss_weight_eff": loss_pack.get("phys_loss_weight_eff"),
        "consistency_loss_weight_eff": loss_pack.get("consistency_loss_weight_eff"),
        "phys_gate_enabled": loss_pack.get("phys_gate_enabled"),
        "phys_gate_passed": loss_pack.get("phys_gate_passed"),
        "phys_gate_reason": loss_pack.get("phys_gate_reason"),
        "lr": float(opt.param_groups[0]["lr"]) if opt.param_groups else None,
    }
    return append_train_metric_jsonl(rec, path)
