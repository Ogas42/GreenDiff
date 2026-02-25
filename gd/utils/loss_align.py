from typing import Tuple

import torch
import torch.nn.functional as F


def _flatten_canonical_sublattice(x: torch.Tensor):
    """
    Support canonical sublattice-resolved LDOS `(B,K,2,H,W)` in alignment helpers by
    flattening to `(B*2,K,H,W)` without mixing the energy axis across sublattices.
    """
    if x.dim() == 5:
        if x.shape[2] != 2:
            raise ValueError(f"Expected canonical sublattice shape (B,K,2,H,W), got {tuple(x.shape)}")
        b, k, s, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * s, k, h, w)
        return flat, (b, k, s, h, w)
    return x, None


def _restore_canonical_sublattice(x: torch.Tensor, shape_meta):
    if shape_meta is None:
        return x
    b, k, s, h, w = shape_meta
    return x.reshape(b, s, k, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _restore_per_energy_loss(loss: torch.Tensor, shape_meta):
    if shape_meta is None:
        return loss
    b, k, s, _, _ = shape_meta
    return loss.reshape(b, s, k).mean(dim=1)


def loss_map(
    pred: torch.Tensor,
    obs: torch.Tensor,
    *,
    loss_type: str = "mse",
    huber_beta: float = 0.1,
    log_cosh_eps: float = 1.0e-6,
) -> torch.Tensor:
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, obs, reduction="none", beta=huber_beta)
    if loss_type == "l1":
        return (pred - obs).abs()
    if loss_type == "log_cosh":
        return torch.log(torch.cosh(pred - obs) + log_cosh_eps)
    return (pred - obs) ** 2


def per_energy_affine(pred: torch.Tensor, obs: torch.Tensor, *, eps: float = 1.0e-6) -> torch.Tensor:
    pred_f, pred_meta = _flatten_canonical_sublattice(pred)
    obs_f, obs_meta = _flatten_canonical_sublattice(obs)
    if pred_meta != obs_meta:
        raise ValueError(f"per_energy_affine shape mismatch: pred={tuple(pred.shape)} obs={tuple(obs.shape)}")
    pred = pred_f
    obs = obs_f
    pred_mean = pred.mean(dim=(2, 3), keepdim=True)
    obs_mean = obs.mean(dim=(2, 3), keepdim=True)
    pred_var = ((pred - pred_mean) ** 2).mean(dim=(2, 3), keepdim=True).clamp_min(eps)
    cov = ((pred - pred_mean) * (obs - obs_mean)).mean(dim=(2, 3), keepdim=True)
    a = cov / pred_var
    b = obs_mean - a * pred_mean
    out = a * pred + b
    return _restore_canonical_sublattice(out, pred_meta)


def align_pred(
    pred: torch.Tensor,
    obs: torch.Tensor,
    *,
    enabled: bool,
    max_shift: int,
    loss_type: str = "mse",
    huber_beta: float = 0.1,
    log_cosh_eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_f, pred_meta = _flatten_canonical_sublattice(pred)
    obs_f, obs_meta = _flatten_canonical_sublattice(obs)
    if pred_meta != obs_meta:
        raise ValueError(f"align_pred shape mismatch: pred={tuple(pred.shape)} obs={tuple(obs.shape)}")
    pred = pred_f
    obs = obs_f
    if not enabled or max_shift <= 0:
        cur = loss_map(
            pred,
            obs,
            loss_type=loss_type,
            huber_beta=huber_beta,
            log_cosh_eps=log_cosh_eps,
        ).mean(dim=(2, 3))
        return _restore_canonical_sublattice(pred, pred_meta), _restore_per_energy_loss(cur, pred_meta)

    best_loss = None
    best_pred = None
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted = torch.roll(pred, shifts=(dx, dy), dims=(2, 3))
            cur = loss_map(
                shifted,
                obs,
                loss_type=loss_type,
                huber_beta=huber_beta,
                log_cosh_eps=log_cosh_eps,
            ).mean(dim=(2, 3))
            if best_loss is None:
                best_loss = cur
                best_pred = shifted
                continue
            mask = cur < best_loss
            best_loss = torch.where(mask, cur, best_loss)
            best_pred = torch.where(mask.unsqueeze(-1).unsqueeze(-1), shifted, best_pred)
    return _restore_canonical_sublattice(best_pred, pred_meta), _restore_per_energy_loss(best_loss, pred_meta)
