import copy
import os
import sys

import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.models.latent_green import LatentGreen  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402


def _small_cfg(backbone: str, latent_downsample: int = 2, use_coord_grid: bool = True):
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["data"]["resolution"] = 8
    cfg["data"]["K"] = 4
    cfg["data"]["energies"] = {"mode": "linspace", "Emin": -0.2, "Emax": 0.2, "list": []}
    cfg["physics"]["hamiltonian"]["t"] = 2.7
    cfg["vae"]["latent_downsample"] = latent_downsample
    cfg["vae"]["latent_channels"] = 4
    cfg["latent_green"]["conditioning"]["use_physics_meta"] = True
    cfg["latent_green"]["conditioning"]["embed_dim"] = 16
    cfg["latent_green"]["model"]["backbone"] = backbone
    cfg["latent_green"]["model"]["base_channels"] = 16
    cfg["latent_green"]["model"]["hidden_channels"] = 16
    cfg["latent_green"]["model"]["num_res_blocks"] = 1
    cfg["latent_green"]["model"]["dropout"] = 0.0
    cfg["latent_green"]["model"]["time_embed_dim"] = 32
    cfg["latent_green"]["model"]["fno_layers"] = 2
    cfg["latent_green"]["model"]["fno_modes_x"] = 16
    cfg["latent_green"]["model"]["fno_modes_y"] = 16
    cfg["latent_green"]["model"]["use_coord_grid"] = use_coord_grid
    cfg["latent_green"]["model"]["local_branch_channels"] = 16
    cfg["latent_green"]["model"]["local_branch_depth"] = 2
    return cfg


def _physics_meta(cfg, batch_size: int):
    return {"hopping": torch.full((batch_size,), float(cfg["physics"]["hamiltonian"]["t"]))}


def test_latent_green_backbones_forward_contract():
    for backbone in ("cnn", "fno", "hybrid_fno"):
        cfg = _small_cfg(backbone)
        model = LatentGreen(cfg).eval()
        latent_res = cfg["data"]["resolution"] // cfg["vae"]["latent_downsample"]
        z = torch.randn(2, cfg["vae"]["latent_channels"], latent_res, latent_res)
        t = torch.zeros(2, dtype=torch.long)
        physics_meta = _physics_meta(cfg, z.shape[0])

        g_pred = model(z, t, physics_meta=physics_meta)
        assert g_pred.shape == (z.shape[0], model.obs_channels, cfg["data"]["resolution"], cfg["data"]["resolution"])
        assert torch.isfinite(g_pred).all()
        assert torch.all(g_pred >= 0.0)

        aux = model(z, t, physics_meta=physics_meta, return_aux=True)
        assert set(aux.keys()) == {"ldos_lin", "psi_real", "psi_imag", "src"}
        assert aux["ldos_lin"].shape == g_pred.shape
        assert aux["psi_real"].shape == aux["psi_imag"].shape == (z.shape[0], model.obs_channels, cfg["data"]["resolution"], cfg["data"]["resolution"])
        assert aux["src"].shape == (z.shape[0], model.obs_channels, cfg["data"]["resolution"], cfg["data"]["resolution"])

        g_pred_psi, psi_r, psi_i, src = model(z, t, physics_meta=physics_meta, return_psi=True)
        assert g_pred_psi.shape == g_pred.shape
        assert psi_r.shape == psi_i.shape == aux["psi_real"].shape
        assert src.shape == aux["src"].shape


def test_latent_green_fno_backbones_cover_downsample_and_coord_toggle():
    for backbone in ("fno", "hybrid_fno"):
        for latent_downsample in (1, 2, 4):
            use_coord_grid = not (backbone == "fno" and latent_downsample == 1)
            cfg = _small_cfg(backbone, latent_downsample=latent_downsample, use_coord_grid=use_coord_grid)
            model = LatentGreen(cfg).eval()
            latent_res = cfg["data"]["resolution"] // cfg["vae"]["latent_downsample"]
            z = torch.randn(1, cfg["vae"]["latent_channels"], latent_res, latent_res)
            g_pred = model(z)
            assert g_pred.shape[-2:] == (cfg["data"]["resolution"], cfg["data"]["resolution"])
            assert torch.isfinite(g_pred).all()

