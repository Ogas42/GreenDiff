import os
import sys

import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.data.structural_defect_sampler import StructuralDefectSampler  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402


def test_structural_defect_sampler_shapes_and_reproducibility():
    cfg = load_config("configs/default.yaml")
    sampler = StructuralDefectSampler(cfg["potential_sampler"]["structural"])
    a = sampler.sample_graphene(8, 10, seed=123)
    b = sampler.sample_graphene(8, 10, seed=123)

    assert set(a.keys()) == {"vacancy_mask", "onsite_ab_delta", "bond_mod"}
    assert a["vacancy_mask"].shape == (2, 8, 10)
    assert a["onsite_ab_delta"].shape == (2, 8, 10)
    assert a["bond_mod"].shape == (3, 8, 10)
    assert a["vacancy_mask"].dtype == torch.bool
    assert a["onsite_ab_delta"].dtype == torch.float32
    assert a["bond_mod"].dtype == torch.float32
    for k in a:
        assert torch.equal(a[k], b[k])


def test_structural_defect_sampler_can_emit_missing_bonds():
    cfg = load_config("configs/default.yaml")
    scfg = dict(cfg["potential_sampler"]["structural"])
    scfg["enabled"] = True
    scfg["family"] = "bond_disorder"
    scfg["bond_disorder"] = {
        "delta_range": [0.1, 0.1],
        "missing_bond_prob": 1.0,
        "apply_prob": 1.0,
    }
    sampler = StructuralDefectSampler(scfg)
    out = sampler.sample_graphene(6, 6, seed=7)
    assert (out["bond_mod"] == -1.0).any()


def test_sublattice_selective_is_spatially_varying_and_ab_opposite():
    scfg = {
        "enabled": True,
        "family": "sublattice_selective",
        "sublattice_selective": {
            "amplitude_range": [0.2, 0.2],
            "mode": "ab_opposite",
            "correlation_frac_range": [0.15, 0.15],
            "localized_envelope_prob": 1.0,
            "localized_quantile": 0.5,
        },
    }
    sampler = StructuralDefectSampler(scfg)
    out = sampler.sample_graphene(24, 24, seed=11)
    a = out["onsite_ab_delta"][0]
    b = out["onsite_ab_delta"][1]
    assert float(a.std().item()) > 1.0e-4
    assert float((a + b).abs().max().item()) < 1.0e-5
