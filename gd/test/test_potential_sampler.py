import copy
import os
import sys

import torch


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.data.potential_sampler import PotentialSampler  # noqa: E402
from gd.utils.config_utils import load_config  # noqa: E402


def _cfg_multiscale():
    cfg = copy.deepcopy(load_config("configs/default.yaml"))
    pcfg = cfg["potential_sampler"]
    pcfg["family"] = "correlated_noise"
    pcfg["normalize"] = False
    pcfg["correlated_noise"] = {
        "corr_length_range": [6.0, 12.0],
        "amplitude_range": [-0.5, 0.5],
        "mode": "multiscale",
        "octaves": [1, 2, 4],
        "octave_amplitude_power": 1.0,
        "global_bias_range": [0.0, 0.0],
    }
    return pcfg


def test_correlated_noise_multiscale_reproducible_and_finite():
    sampler = PotentialSampler(_cfg_multiscale())
    a = sampler.sample(32, 32, None, 123)
    b = sampler.sample(32, 32, None, 123)
    assert a.shape == (32, 32)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)


def test_correlated_noise_multiscale_is_continuous_field_not_piecewise():
    sampler = PotentialSampler(_cfg_multiscale())
    v = sampler.sample(64, 64, None, 7)
    # Continuous-valued field should have many unique values (far from domain-wall style regions).
    unique_count = torch.unique(torch.round(v * 1000)).numel()
    assert int(unique_count) > 200
    # And spatial gradients should be widely nonzero.
    gx = (v[:, 1:] - v[:, :-1]).abs()
    gy = (v[1:, :] - v[:-1, :]).abs()
    nonzero_grad_frac = torch.cat([gx.reshape(-1), gy.reshape(-1)]).gt(1.0e-4).float().mean()
    assert float(nonzero_grad_frac) > 0.7
    # Smoke target range for puddles-style preset.
    assert 0.02 <= float(v.std()) <= 0.5
