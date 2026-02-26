import os
import sys


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.utils.config_utils import load_config  # noqa: E402
from gd.utils.obs_layout import cache_ldos_schema_metadata  # noqa: E402


def test_graphene_fixed_t_phys_fft_longrun_config_smoke():
    cfg = load_config("configs/graphene_fixed_t_phys_fft.yaml")

    # Fixed-t graphene physical baseline with FFT/PSD as strong auxiliary constraints.
    assert cfg["physics"]["hamiltonian"]["type"] == "graphene"
    assert float(cfg["physics"]["hamiltonian"]["t"]) == 2.7

    data_cfg = cfg["data"]
    assert int(data_cfg["num_samples_total"]) == 50000
    assert int(data_cfg["cache_shard_size"]) == 256
    assert int(data_cfg["num_workers"]) > 0
    assert bool(data_cfg["persistent_workers"]) is True

    vae_train = cfg["vae"]["training"]
    green_model = cfg["latent_green"]["model"]
    green_train = cfg["latent_green"]["training"]
    diff_train = cfg["diffusion"]["training"]

    assert int(vae_train["max_steps"]) == 50000
    assert int(green_train["max_steps"]) == 100000
    assert int(diff_train["max_steps"]) == 200000

    assert green_model["data_loss_domain"] == "linear_normalized"
    assert bool(green_model["use_fft_loss"]) is True
    assert float(green_model["fft_loss_weight"]) > 0.0
    assert float(green_model["psd_loss_weight"]) > 0.0
    assert int(green_model["aux_warmup_steps"]) == 5000
    assert bool(green_model["peak_control"]["enabled"]) is True
    assert float(green_model["peak_control"]["topk_loss_weight"]) > 0.0
    assert float(green_model["peak_control"]["peak_ratio_penalty_weight"]) > 0.0

    assert bool(green_train["lr_schedule"]["enabled"]) is True
    assert int(green_train["lr_schedule"]["warmup_steps"]) == 5000
    assert bool(diff_train["lr_schedule"]["enabled"]) is True
    assert int(diff_train["lr_schedule"]["warmup_steps"]) == 10000
    assert diff_train["inverse_target"] == "v_only"
    assert bool(diff_train["allow_unclosed_inverse"]) is True
    assert bool(diff_train["phys_supervision"]["enabled"]) is True
    assert diff_train["phys_supervision"]["domain"] == "linear_normalized"
    assert bool(diff_train["phys_gate"]["enabled"]) is True

    meta = cache_ldos_schema_metadata(cfg)
    assert "potential_normalize" in meta
    assert bool(meta["potential_normalize"]) is False
