import copy
import os
import sys

import pytest
import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.data.kpm_forward import KPMForward  # noqa: E402
from gd.data.structural_defect_sampler import StructuralDefectSampler  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402


def _kpm_cfg_small():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    cfg["data"]["resolution"] = 6
    cfg["data"]["K"] = 3
    cfg["data"]["energies"] = {"mode": "linspace", "Emin": -0.15, "Emax": 0.15, "list": []}
    cfg["physics"]["hamiltonian"]["t"] = 2.7
    cfg["physics"]["kpm"]["direct_inverse"] = {"enabled": True, "max_sites": 4096}
    return cfg


def test_kpm_forward_graphene_defects_smoke():
    pytest.importorskip("kwant")
    cfg = _kpm_cfg_small()
    kpm = KPMForward(
        {
            "kpm": cfg["physics"]["kpm"],
            "hamiltonian": cfg["physics"]["hamiltonian"],
            "data": cfg["data"],
            "rng_seed": 0,
        }
    )
    H = W = cfg["data"]["resolution"]
    V = torch.zeros(H, W)
    defect_meta = StructuralDefectSampler(cfg["potential_sampler"]["structural"]).sample_graphene(H, W, seed=0)
    out = kpm.compute_ldos(V, torch.linspace(-0.1, 0.1, cfg["data"]["K"]).tolist(), defect_meta=defect_meta)
    assert out.shape == (cfg["data"]["K"], 2, H, W)
    assert torch.isfinite(out).all()
