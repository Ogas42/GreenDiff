import copy
import os
import sys

import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.data.structural_defect_sampler import StructuralDefectSampler  # noqa: E402
from gd.models.latent_green import LatentGreen  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402
from gd.utils.ldos_transform import ldos_obs_from_linear  # noqa: E402


def _small_cfg():
    cfg = load_config("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["data"]["resolution"] = 8
    cfg["data"]["K"] = 4
    cfg["data"]["energies"] = {"mode": "linspace", "Emin": -0.2, "Emax": 0.2, "list": []}
    cfg["physics"]["hamiltonian"]["t"] = 2.7
    cfg["vae"]["latent_downsample"] = 2
    cfg["vae"]["latent_channels"] = 4
    cfg["latent_green"]["model"]["backbone"] = "cnn"
    cfg["latent_green"]["model"]["base_channels"] = 16
    cfg["latent_green"]["model"]["num_res_blocks"] = 1
    cfg["latent_green"]["model"]["use_fft_loss"] = False
    cfg["latent_green"]["model"]["psd_loss_weight"] = 0.0
    cfg["latent_green"]["model"]["multiscale_loss_weight"] = 0.0
    cfg["latent_green"]["model"]["data_loss_domain"] = "linear_normalized"
    cfg["latent_green"]["model"]["loss_type"] = "huber"
    cfg["latent_green"]["conditioning"]["use_physics_meta"] = True
    return cfg


def test_graphene_residual_operator_fixed_t_nonzero_and_masked():
    cfg = _small_cfg()
    model = LatentGreen(cfg).eval()
    B, H, W = 2, cfg["data"]["resolution"], cfg["data"]["resolution"]
    z = torch.randn(B, cfg["vae"]["latent_channels"], H // 2, W // 2)
    V = torch.randn(B, 1, H, W)
    physics_meta = {"hopping": torch.full((B,), 2.7), "eta": torch.full((B,), 0.01)}
    defect = StructuralDefectSampler(cfg["potential_sampler"]["structural"]).sample_graphene(H, W, seed=1)
    defect_meta = {k: v.unsqueeze(0).repeat(B, *([1] * v.dim())) for k, v in defect.items()}

    g_pred, psi_r, psi_i, src = model(
        z, torch.zeros(B, dtype=torch.long), physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True
    )
    res = model.residual_loss(psi_r, psi_i, src, V, physics_meta=physics_meta, defect_meta=defect_meta)
    aux = model.get_last_residual_aux()
    assert torch.isfinite(res)
    assert float(res.detach()) > 0.0
    assert 0.0 <= float(aux.get("residual_active_frac", -1.0)) <= 1.0

    losses = model.loss(g_pred, ldos_obs_from_linear(torch.zeros(B, cfg["data"]["K"], 2, H, W), cfg["data"]), residual_loss=res)
    assert torch.isfinite(losses["sum_rule_loss"])


def test_linear_normalized_data_loss_ignores_align_and_affine_flags():
    cfg = _small_cfg()
    cfg_a = copy.deepcopy(cfg)
    cfg_b = copy.deepcopy(cfg)
    cfg_b["latent_green"]["model"]["per_energy_affine"] = True
    cfg_b["latent_green"]["model"]["energy_align"] = {"enabled": True, "max_shift": 2}

    m_a = LatentGreen(cfg_a).eval()
    m_b = LatentGreen(cfg_b).eval()
    B, K, H, W = 2, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"]
    g_pred = torch.rand(B, 2 * K, H, W)
    g_obs_lin = torch.rand(B, K, 2, H, W)
    g_obs = ldos_obs_from_linear(g_obs_lin, cfg["data"])

    la = m_a.loss(g_pred, g_obs)["data_loss"]
    lb = m_b.loss(g_pred, g_obs)["data_loss"]
    assert torch.allclose(la, lb, atol=1.0e-6, rtol=1.0e-6)


def test_peak_control_losses_penalize_peak_overshoot():
    cfg = _small_cfg()
    cfg["latent_green"]["model"]["peak_control"] = {
        "enabled": True,
        "log_aux_weight": 0.10,
        "log_aux_huber_beta": 0.1,
        "log_aux_scale": "p95_obs_per_sample",
        "topk_loss_weight": 0.20,
        "topk_frac": 0.05,
        "topk_huber_beta": 0.1,
        "peak_ratio_penalty_weight": 0.05,
        "peak_ratio_cap": 2.0,
    }
    m = LatentGreen(cfg).eval()
    B, K, H, W = 2, cfg["data"]["K"], cfg["data"]["resolution"], cfg["data"]["resolution"]
    g_obs_lin = torch.full((B, K, 2, H, W), 0.05)
    g_obs_lin[:, :, :, 2, 2] = 0.2
    g_obs = ldos_obs_from_linear(g_obs_lin, cfg["data"])

    pred_lin_c = g_obs_lin.clone()
    pred_lin_c[:, :, :, 2, 2] = 1.5
    g_pred = pred_lin_c.permute(0, 2, 1, 3, 4).reshape(B, 2 * K, H, W)

    losses = m.loss(g_pred, g_obs)
    assert float(losses["log_aux_loss"].item()) > 0.0
    assert float(losses["topk_peak_loss"].item()) > 0.0
    assert float(losses["peak_ratio_penalty"].item()) > 0.0
