import os
import sys
import json

import pytest


def _ensure_project_root():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.trainers.diffusion_trainer import DiffusionTrainer  # noqa: E402


def test_v_only_with_structural_enabled_is_rejected_by_default():
    cfg = {
        "potential_sampler": {
            "structural": {"enabled": True},
        }
    }
    train_cfg = {
        "inverse_target": "v_only",
        "allow_unclosed_inverse": False,
    }
    with pytest.raises(ValueError, match="unclosed"):
        DiffusionTrainer._validate_inverse_target_cfg(cfg, train_cfg)


def test_v_only_with_structural_enabled_can_override():
    cfg = {
        "potential_sampler": {
            "structural": {"enabled": True},
        }
    }
    train_cfg = {
        "inverse_target": "v_only",
        "allow_unclosed_inverse": True,
    }
    DiffusionTrainer._validate_inverse_target_cfg(cfg, train_cfg)


def test_diffusion_trainer_cfg_bool_parses_falsey_strings():
    assert DiffusionTrainer._cfg_bool(False, default=True) is False
    assert DiffusionTrainer._cfg_bool("false", default=True) is False
    assert DiffusionTrainer._cfg_bool("0", default=True) is False
    assert DiffusionTrainer._cfg_bool("off", default=True) is False
    assert DiffusionTrainer._cfg_bool("true", default=False) is True


def test_phys_gate_state_reads_green_metrics_and_fails_thresholds(tmp_path):
    metrics_path = tmp_path / "green_eval_val.json"
    payload = {
        "metrics": {
            "rel_l2": 0.72,
            "peak_ratio_mean": 8.5,
            "pred_obs_mean_ratio_mean": 0.95,
        }
    }
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = {"paths": {"workdir": str(tmp_path), "runs_root": str(tmp_path)}}
    train_cfg = {
        "phys_gate": {
            "enabled": True,
            "allow_override": False,
            "green_metrics_path": str(metrics_path),
            "rel_l2_max": 0.60,
            "peak_ratio_max": 5.0,
            "mean_ratio_min": 0.70,
            "mean_ratio_max": 1.30,
        }
    }
    state = DiffusionTrainer._resolve_phys_gate_state(cfg, train_cfg, is_main=False)
    assert state["enabled"] is True
    assert state["passed"] is False
    assert "green_gate_fail" in str(state.get("reason"))


def test_phys_gate_state_reads_green_metrics_and_passes(tmp_path):
    metrics_path = tmp_path / "green_eval_val.json"
    payload = {
        "metrics": {
            "rel_l2": 0.42,
            "peak_ratio_mean": 3.1,
            "pred_obs_mean_ratio_mean": 1.02,
        }
    }
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = {"paths": {"workdir": str(tmp_path), "runs_root": str(tmp_path)}}
    train_cfg = {
        "phys_gate": {
            "enabled": True,
            "allow_override": False,
            "green_metrics_path": str(metrics_path),
            "rel_l2_max": 0.60,
            "peak_ratio_max": 5.0,
            "mean_ratio_min": 0.70,
            "mean_ratio_max": 1.30,
        }
    }
    state = DiffusionTrainer._resolve_phys_gate_state(cfg, train_cfg, is_main=False)
    assert state["enabled"] is True
    assert state["passed"] is True
    assert state["reason"] == "green_gate_pass"
