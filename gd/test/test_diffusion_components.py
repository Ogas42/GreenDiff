import os
import sys

import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.trainers.diffusion_components import compute_physics_terms  # noqa: E402
from gd.utils.ldos_transform import ldos_obs_from_linear  # noqa: E402


class _IdentityLatentGreen:
    def __call__(self, x0_pred, t_zeros):
        # Returns linear-domain LDOS directly in model/eval space.
        return x0_pred


def _data_cfg_with_scale(scale: float):
    return {
        "K": 4,
        "sublattice_resolved_ldos": False,
        "ldos_transform": {
            "enabled": True,
            "apply_to_cache": True,
            "log": {"enabled": False, "eps": 1.0e-6},
            "quantile": {"enabled": False, "eps": 1.0e-6},
            "scale": float(scale),
            "cache_scaled": True,
        },
    }


def _train_cfg_linear_norm():
    return {
        "phys_loss_type": "mse",
        "huber_beta": 0.1,
        "per_energy_affine": False,
        "energy_align": {"enabled": False, "max_shift": 0},
        "log_cosh_eps": 1.0e-6,
        "psd_loss_weight": 0.0,
        "psd_eps": 1.0e-8,
        "consistency_loss_weight": 0.0,
        "topk_phys": {"enabled": False, "k": 0},
        "energy_weights": [],
        "energy_weight_mode": "uniform",
        "energy_weight_eps": 1.0e-6,
        "energy_weight_power": 1.0,
        "phys_loss_weight": 0.0,
        "phys_warmup": {"enabled": False, "warmup_steps": 0},
        "phys_supervision": {
            "enabled": True,
            "domain": "linear_normalized",
            "monitor_when_disabled": True,
            "normalize_per_sample_rms": True,
            "consistency_on_normalized_linear": True,
        },
    }


def test_physics_terms_linear_normalized_invariant_to_ldos_scale():
    torch.manual_seed(0)
    B, K, H, W = 2, 4, 8, 8
    pred_lin = torch.rand(B, K, H, W) * 0.4 + 0.05
    obs_lin = pred_lin * 0.9 + 0.01 * torch.rand(B, K, H, W)

    data_cfg_1 = _data_cfg_with_scale(1.0e3)
    data_cfg_2 = _data_cfg_with_scale(1.0e5)
    g_obs_1 = ldos_obs_from_linear(obs_lin, data_cfg_1)
    g_obs_2 = ldos_obs_from_linear(obs_lin, data_cfg_2)
    train_cfg = _train_cfg_linear_norm()
    alpha_t = torch.ones(B, 1, 1, 1)
    lg = _IdentityLatentGreen()

    out1 = compute_physics_terms(
        x0_pred=pred_lin,
        latent_green=lg,
        g_obs=g_obs_1,
        alpha_t=alpha_t,
        train_cfg=train_cfg,
        data_cfg=data_cfg_1,
        compute_consistency=True,
    )
    out2 = compute_physics_terms(
        x0_pred=pred_lin,
        latent_green=lg,
        g_obs=g_obs_2,
        alpha_t=alpha_t,
        train_cfg=train_cfg,
        data_cfg=data_cfg_2,
        compute_consistency=True,
    )

    assert out1["phys_eval_domain"] == "linear_normalized"
    assert out2["phys_eval_domain"] == "linear_normalized"
    assert out1["raw_phys_loss"] is not None and out2["raw_phys_loss"] is not None
    assert out1["raw_consistency_loss"] is not None and out2["raw_consistency_loss"] is not None
    assert abs(float(out1["raw_phys_loss"]) - float(out2["raw_phys_loss"])) < 1.0e-6
    assert abs(float(out1["raw_consistency_loss"]) - float(out2["raw_consistency_loss"])) < 1.0e-6

