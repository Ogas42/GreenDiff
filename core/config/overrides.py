from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


def deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def apply_dotted_overrides(config: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        deep_set(out, key, value)
    return out


def profile_overrides(profile: str) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {
        "none": {},
        "local_4060": {
            "project.device": "cuda",
            "project.precision": "fp16",
            "project.compile": False,
            "project.cudnn_benchmark": False,
            "data.num_workers": 2,
            "data.shard_workers": 2,
            "data.pin_memory": True,
            "data.persistent_workers": False,
            "data.prefetch_factor": 2,
            "vae.training.batch_size": 2,
            "vae.training.max_steps": 500,
            "vae.training.log_every": 20,
            "vae.training.ckpt_every": 100,
            "latent_green.training.batch_size": 2,
            "latent_green.training.max_steps": 1000,
            "latent_green.training.log_every": 20,
            "latent_green.training.ckpt_every": 200,
            "latent_green.training.lr_schedule.enabled": False,
            "diffusion.training.batch_size": 1,
            "diffusion.training.max_steps": 2000,
            "student.training.batch_size": 2,
            "student.training.max_steps": 2000,
            "validation.enabled": False,
        },
        "local_4060_smoke": {
            "project.device": "cuda",
            "project.precision": "fp16",
            "project.compile": False,
            "data.num_workers": 0,
            "data.shard_workers": 0,
            "data.pin_memory": False,
            "vae.training.batch_size": 1,
            "vae.training.max_steps": 10,
            "vae.training.log_every": 1,
            "vae.training.ckpt_every": 10,
            "latent_green.training.batch_size": 1,
            "latent_green.training.max_steps": 10,
            "latent_green.training.log_every": 1,
            "latent_green.training.ckpt_every": 10,
            "diffusion.training.batch_size": 1,
            "diffusion.training.max_steps": 10,
            "student.training.batch_size": 1,
            "student.training.max_steps": 10,
            "validation.enabled": False,
        },
        "remote_a6000": {
            "project.device": "cuda",
            "project.precision": "bf16",
            "project.compile": True,
            "project.cudnn_benchmark": True,
            "data.num_workers": 8,
            "data.shard_workers": 8,
            "data.pin_memory": True,
            "data.persistent_workers": True,
            "data.prefetch_factor": 4,
            "vae.training.batch_size": 32,
            "vae.training.max_steps": 30000,
            "vae.training.log_every": 200,
            "vae.training.ckpt_every": 2000,
            "latent_green.training.batch_size": 32,
            "latent_green.training.max_steps": 40000,
            "latent_green.training.log_every": 100,
            "latent_green.training.ckpt_every": 2000,
            "latent_green.training.lr_schedule.enabled": True,
        },
    }
    if profile not in profiles:
        raise KeyError(f"Unknown profile '{profile}'. Available: {sorted(profiles)}")
    return profiles[profile]


def apply_profile(config: Dict[str, Any], profile: str | None) -> Dict[str, Any]:
    if not profile or profile == "none":
        return config
    return apply_dotted_overrides(config, profile_overrides(profile))

