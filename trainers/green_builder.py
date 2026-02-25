from __future__ import annotations

import os
import re
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim

from gd.core.checkpoints.manager import CheckpointManager
from gd.core.typing.types import ResumeState


def ckpt_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"_step_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def build_frozen_vae(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from gd.models.vae import VAE

    vae = VAE(cfg).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def build_green_model(
    cfg: Dict[str, Any],
    device: torch.device,
    dist_ctx: Any,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    from gd.models.latent_green import LatentGreen

    model = LatentGreen(cfg).to(device)
    if dist_ctx.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if device.type == "cuda" else None,
        )
    model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    return model, model_core


def build_green_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])


def prepare_green_train_metrics_path(workdir: str) -> str:
    metrics_dir = os.path.join(workdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    return os.path.join(metrics_dir, "green_train_metrics.jsonl")


def load_vae_checkpoint(
    *,
    ckpt_mgr: CheckpointManager,
    vae: torch.nn.Module,
    device: torch.device,
    is_main: bool,
) -> str | None:
    vae_ckpt = ckpt_mgr.find_latest("vae_step_*.pt")
    if not vae_ckpt:
        if is_main:
            print("Warning: No VAE checkpoint found! Training on random latents.")
        return None

    if is_main:
        print(f"Loading VAE from {vae_ckpt}")
    state = ckpt_mgr.load_state_dict(vae_ckpt, map_location=device, normalize=True, torch_module=torch)
    vae.load_state_dict(state)
    return vae_ckpt


def resume_green_model(
    *,
    ckpt_mgr: CheckpointManager,
    model_core: torch.nn.Module,
    device: torch.device,
    is_main: bool,
) -> ResumeState:
    lg_ckpt = ckpt_mgr.find_latest("latent_green_step_*.pt")
    if not lg_ckpt:
        if is_main:
            print("No Latent Green checkpoint found. Starting from scratch.")
        return ResumeState(step=0, checkpoint_path=None)

    try:
        state = ckpt_mgr.load_state_dict(lg_ckpt, map_location=device, normalize=True, torch_module=torch)
        model_core.load_state_dict(state)
        step = ckpt_step_from_path(lg_ckpt) or 0
        if is_main:
            print(f"Resuming Latent Green from {lg_ckpt} (step={step})")
        return ResumeState(step=step, checkpoint_path=lg_ckpt)
    except RuntimeError as exc:
        if is_main:
            print(f"Warning: failed to resume Latent Green checkpoint ({exc}); restarting from scratch.")
        return ResumeState(step=0, checkpoint_path=None, meta={"resume_error": str(exc)})
