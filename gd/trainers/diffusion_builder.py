from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim as optim

from gd.core.checkpoints.manager import CheckpointManager
from gd.core.typing.types import ResumeState


def ckpt_step_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"_step_(\d+)(?:_ema)?\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _save_target(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "_orig_mod"):
        return module._orig_mod  # type: ignore[attr-defined]
    return module


def _ema_path_for_diffusion_ckpt(diff_ckpt_path: str | None) -> str | None:
    if not diff_ckpt_path:
        return None
    base, ext = os.path.splitext(diff_ckpt_path)
    return f"{base}_ema{ext}"


def build_frozen_vae(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from gd.models.vae import VAE

    vae = VAE(cfg).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def build_frozen_latent_green(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    from gd.models.latent_green import LatentGreen

    latent_green = LatentGreen(cfg).to(device)
    latent_green.eval()
    for p in latent_green.parameters():
        p.requires_grad = False
    return latent_green


def build_diffusion_model(
    cfg: Dict[str, Any],
    device: torch.device,
    dist_ctx: Any,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    from gd.models.diffusion import LatentDiffusion

    model = LatentDiffusion(cfg).to(device)
    if dist_ctx.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if device.type == "cuda" else None,
        )
    model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    return model, model_core


def build_diffusion_optimizer(model: torch.nn.Module, train_cfg: Dict[str, Any]) -> optim.Optimizer:
    return optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))


def build_ema_model_if_enabled(
    *,
    model_core: torch.nn.Module,
    train_cfg: Dict[str, Any],
    device: torch.device,
    is_main: bool,
) -> Tuple[Optional[torch.nn.Module], Optional[float]]:
    ema_cfg = train_cfg.get("ema", {})
    use_ema = bool(ema_cfg.get("enabled", False))
    if not use_ema or not is_main:
        return None, None
    ema_model = copy.deepcopy(_save_target(model_core)).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = float(ema_cfg.get("decay", 0.9999))
    return ema_model, ema_decay


def prepare_diffusion_train_metrics_path(workdir: str) -> str:
    metrics_dir = os.path.join(workdir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    return os.path.join(metrics_dir, "diffusion_train_metrics.jsonl")


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
            print("Warning: No VAE checkpoint found! Training diffusion on random latents.")
        return None
    if is_main:
        print(f"Loading VAE from {vae_ckpt}")
    state = ckpt_mgr.load_state_dict(vae_ckpt, map_location=device, normalize=True, torch_module=torch)
    vae.load_state_dict(state)
    return vae_ckpt


def load_latent_green_checkpoint(
    *,
    ckpt_mgr: CheckpointManager,
    latent_green: torch.nn.Module,
    model_core: torch.nn.Module | None,
    cfg: Dict[str, Any],
    device: torch.device,
    is_main: bool,
) -> str | None:
    lg_ckpt = ckpt_mgr.find_latest("latent_green_step_*.pt")
    if not lg_ckpt:
        if is_main:
            print("Warning: No Latent Green checkpoint found!")
        return None
    if is_main:
        print(f"Loading Latent Green from {lg_ckpt}")
    state = ckpt_mgr.load_state_dict(lg_ckpt, map_location=device, normalize=True, torch_module=torch)
    try:
        latent_green.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
            "Re-train the Green stage with schema-v2 cache."
        ) from exc
    use_green_attn = bool(cfg.get("diffusion", {}).get("model", {}).get("use_green_attention", False))
    if use_green_attn and model_core is not None and hasattr(model_core, "latent_green"):
        try:
            model_core.latent_green.load_state_dict(state)
        except Exception as exc:
            if is_main:
                print(f"Warning: failed to load latent_green into diffusion model attention branch: {exc}")
    return lg_ckpt


def resume_diffusion_model(
    *,
    ckpt_mgr: CheckpointManager,
    model_core: torch.nn.Module,
    device: torch.device,
    is_main: bool,
) -> ResumeState:
    diff_ckpt = ckpt_mgr.find_latest_in_current("diffusion_step_*.pt") or ckpt_mgr.find_latest("diffusion_step_*.pt")
    if not diff_ckpt:
        if is_main:
            print("No Diffusion checkpoint found. Starting from scratch.")
        return ResumeState(step=0, checkpoint_path=None)
    try:
        state = ckpt_mgr.load_state_dict(diff_ckpt, map_location=device, normalize=True, torch_module=torch)
        _save_target(model_core).load_state_dict(state)
        step = ckpt_step_from_path(diff_ckpt) or 0
        if is_main:
            print(f"Resuming Diffusion from {diff_ckpt} (step={step})")
        return ResumeState(step=step, checkpoint_path=diff_ckpt)
    except RuntimeError as exc:
        if is_main:
            print(
                "Warning: failed to resume Diffusion checkpoint. This can happen after the Phase-1 "
                "sublattice-resolved LDOS channel upgrade (K -> 2K). "
                f"Original error: {exc}"
            )
        return ResumeState(step=0, checkpoint_path=None, meta={"resume_error": str(exc)})


def resume_diffusion_ema(
    *,
    ema_model: torch.nn.Module | None,
    resume_state: ResumeState,
    ckpt_mgr: CheckpointManager,
    device: torch.device,
    is_main: bool,
) -> str | None:
    if ema_model is None:
        return None
    ema_path = _ema_path_for_diffusion_ckpt(resume_state.checkpoint_path)
    if not ema_path:
        return None
    if not os.path.exists(ema_path):
        if is_main:
            print(f"Warning: EMA checkpoint not found for resumed diffusion checkpoint: {ema_path}")
        return None
    try:
        state = ckpt_mgr.load_state_dict(ema_path, map_location=device, normalize=True, torch_module=torch)
        ema_model.load_state_dict(state)
        if is_main:
            print(f"Loaded Diffusion EMA from {ema_path}")
        return ema_path
    except Exception as exc:
        if is_main:
            print(f"Warning: failed to load Diffusion EMA checkpoint ({exc})")
        return None


def save_diffusion_checkpoint(
    *,
    ckpt_mgr: CheckpointManager,
    model_core: torch.nn.Module,
    step: int,
    torch_module: Any = None,
) -> str:
    return ckpt_mgr.save_state_dict("diffusion", step, _save_target(model_core).state_dict(), torch_module=torch_module or torch)


def save_diffusion_ema_checkpoint(
    *,
    ema_model: torch.nn.Module | None,
    ckpt_mgr: CheckpointManager,
    step: int,
    torch_module: Any = None,
) -> str | None:
    if ema_model is None:
        return None
    os.makedirs(ckpt_mgr.current_ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_mgr.current_ckpt_dir, f"diffusion_step_{step}_ema.pt")
    (torch_module or torch).save(_save_target(ema_model).state_dict(), path)
    return path
