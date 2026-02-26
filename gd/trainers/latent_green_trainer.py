from __future__ import annotations

import sys
import time
from collections import deque
from typing import Any, Dict

import torch

from gd.core.checkpoints.manager import CheckpointManager
from gd.core.data.loader_factory import build_train_dataloader
from gd.core.logging.progress import get_tqdm
from gd.core.logging.results import append_train_metric_jsonl
from gd.core.typing.types import ResumeState, StageResult
from gd.trainers.base import StageTrainer
from gd.trainers.green_builder import build_frozen_vae
from gd.trainers.green_builder import build_green_model
from gd.trainers.green_builder import build_green_optimizer
from gd.trainers.green_builder import ckpt_step_from_path
from gd.trainers.green_builder import load_vae_checkpoint
from gd.trainers.green_builder import prepare_green_train_metrics_path
from gd.trainers.green_builder import resume_green_model
from gd.trainers.green_components import build_green_scheduler
from gd.trainers.green_components import compute_green_loss
from gd.trainers.green_components import green_train_step
from gd.trainers.green_components import log_green_train_status
from gd.utils.ldos_transform import force_linear_ldos_mode


def _to_device_tree(x, device, non_blocking: bool = True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device, non_blocking=non_blocking) for k, v in x.items()}
    return x


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


class LatentGreenTrainer(StageTrainer):
    stage_name = "green"
    requires = ["vae"]

    def build(self, ctx: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        force_linear_ldos_mode(cfg, verbose=ctx.dist.is_main, context="gd.green_trainer")
        ckpt_mgr = CheckpointManager(cfg["paths"]["runs_root"], cfg["paths"]["checkpoints"])
        train_cfg = cfg["latent_green"]["training"]
        device = torch.device(ctx.dist.device)
        dataset, sampler, loader = build_train_dataloader(cfg, train_cfg, ctx.dist, split="train")
        debug_fixed_batch = bool(train_cfg.get("debug_fixed_batch", False))
        fixed_batch = next(iter(loader)) if debug_fixed_batch else None
        if debug_fixed_batch and ctx.dist.is_main:
            print("Debug: using fixed batch for training loop.")

        vae = build_frozen_vae(cfg, device)
        model, model_core = build_green_model(cfg, device, ctx.dist)
        opt = build_green_optimizer(model, train_cfg)
        scheduler = build_green_scheduler(opt, train_cfg)

        train_metrics_jsonl = prepare_green_train_metrics_path(cfg["paths"]["workdir"])

        return {
            "device": device,
            "train_cfg": train_cfg,
            "dataset": dataset,
            "sampler": sampler,
            "loader": loader,
            "fixed_batch": fixed_batch,
            "vae": vae,
            "model": model,
            "model_core": model_core,
            "optimizer": opt,
            "scheduler": scheduler,
            "ckpt_mgr": ckpt_mgr,
            "train_metrics_jsonl": train_metrics_jsonl,
            "last_ckpt": None,
        }

    def resume(self, ctx: Any, cfg: Dict[str, Any], components: Dict[str, Any]) -> ResumeState:
        device = components["device"]
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        vae = components["vae"]
        model_core = components["model_core"]

        load_vae_checkpoint(ckpt_mgr=ckpt_mgr, vae=vae, device=device, is_main=ctx.dist.is_main)
        return resume_green_model(ckpt_mgr=ckpt_mgr, model_core=model_core, device=device, is_main=ctx.dist.is_main)

    def run(self, ctx, cfg, components, resume_state: ResumeState) -> StageResult:
        t0 = time.time()
        device = components["device"]
        train_cfg = components["train_cfg"]
        loader = components["loader"]
        sampler = components["sampler"]
        fixed_batch = components["fixed_batch"]
        vae = components["vae"]
        model = components["model"]
        model_core = components["model_core"]
        opt = components["optimizer"]
        scheduler = components["scheduler"]
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        amp = ctx.amp

        max_steps = int(train_cfg["max_steps"])
        log_every = int(train_cfg["log_every"])
        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        ckpt_every = int(train_cfg.get("ckpt_every", 2000))
        show_progress_bar = _cfg_bool(train_cfg.get("show_progress_bar", True), default=True)
        if not sys.stderr.isatty():
            show_progress_bar = False
        if ctx.dist.is_main:
            print(
                f"[gd.green_trainer] show_progress_bar={show_progress_bar} "
                f"(cfg={train_cfg.get('show_progress_bar', None)!r}, stderr_tty={sys.stderr.isatty()})"
            )
        step = int(resume_state.step)
        start_step = step

        noisy_cfg = cfg["latent_green"]["noisy_latent_training"]
        model_cfg = cfg["latent_green"]["model"]
        data_cfg = cfg.get("data", {})

        tqdm = get_tqdm()
        pbar = (
            tqdm(total=max_steps, initial=step, desc="Training Latent Green", dynamic_ncols=True)
            if (ctx.dist.is_main and show_progress_bar)
            else None
        )
        smooth_window = int(train_cfg.get("log_smooth_window", 50))
        loss_hist = deque(maxlen=max(1, smooth_window))
        data_hist = deque(maxlen=max(1, smooth_window))
        rel_hist = deque(maxlen=max(1, smooth_window))
        last_log_time = time.perf_counter()

        last_scalar: Dict[str, float] = {}
        last_rel_model = None

        while step < max_steps:
            if sampler is not None:
                epoch = step // max(1, len(loader))
                sampler.set_epoch(epoch)
            batch_iter = (fixed_batch,) if fixed_batch is not None else loader
            for batch in batch_iter:
                V = batch["V"].to(device, non_blocking=True)
                if V.dim() == 3:
                    V = V.unsqueeze(1)
                g_obs = batch["g_obs"].to(device, non_blocking=True)
                physics_meta = _to_device_tree(batch.get("physics_meta"), device) if isinstance(batch, dict) else None
                defect_meta = _to_device_tree(batch.get("defect_meta"), device) if isinstance(batch, dict) else None

                with torch.no_grad():
                    z, _, _ = vae.encode(V)

                losses, rel_l2, aux_log = compute_green_loss(
                    cfg=cfg,
                    model=model,
                    model_core=model_core,
                    z=z,
                    V=V,
                    g_obs=g_obs,
                    physics_meta=physics_meta,
                    defect_meta=defect_meta,
                    noisy_cfg=noisy_cfg,
                    model_cfg=model_cfg,
                    data_cfg=data_cfg,
                    step=step,
                    device=device,
                    amp=amp,
                )

                green_train_step(
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    losses=losses,
                    grad_clip=grad_clip,
                    amp=amp,
                )

                last_scalar = {
                    "loss": float(losses["loss"].detach().item()),
                    "data_loss": float(losses["data_loss"].detach().item()),
                    "fft_loss": float(losses.get("fft_loss", torch.zeros((), device=device)).detach().item()),
                    "stats_loss": float(losses.get("stats_loss", torch.zeros((), device=device)).detach().item()),
                    "psd_loss": float(losses.get("psd_loss", torch.zeros((), device=device)).detach().item()),
                    "ms_loss": float(losses.get("ms_loss", torch.zeros((), device=device)).detach().item()),
                    "log_aux_loss": float(losses.get("log_aux_loss", torch.zeros((), device=device)).detach().item()),
                    "topk_peak_loss": float(losses.get("topk_peak_loss", torch.zeros((), device=device)).detach().item()),
                    "peak_ratio_penalty": float(losses.get("peak_ratio_penalty", torch.zeros((), device=device)).detach().item()),
                    "residual_loss": float(losses.get("residual_loss", torch.zeros((), device=device)).detach().item()),
                    "sum_rule_loss": float(losses.get("sum_rule_loss", torch.zeros((), device=device)).detach().item()),
                }
                last_rel_model = float(rel_l2.item())

                if ctx.dist.is_main and step % log_every == 0:
                    now = time.perf_counter()
                    elapsed = max(1.0e-6, now - last_log_time)
                    it_s = float(log_every) / elapsed if step > 0 else 0.0
                    if step == 0:
                        it_s = 0.0
                    samples_s = it_s * float(train_cfg["batch_size"]) * float(ctx.dist.world_size)
                    last_log_time = now
                    log_green_train_status(
                        cfg=cfg,
                        model=model,
                        optimizer=opt,
                        step=step,
                        g_obs=g_obs,
                        losses=losses,
                        rel_l2=rel_l2,
                        aux_log=aux_log,
                        loss_hist=loss_hist,
                        data_hist=data_hist,
                        rel_hist=rel_hist,
                        it_s=it_s,
                        samples_s=samples_s,
                        pbar=pbar,
                    )
                    append_train_metric_jsonl(
                        {
                            "task": "green_train",
                            "stage": "green",
                            "step": int(step),
                            "loss": last_scalar["loss"],
                            "data_loss": last_scalar["data_loss"],
                            "fft_loss": last_scalar["fft_loss"],
                            "stats_loss": last_scalar["stats_loss"],
                            "psd_loss": last_scalar["psd_loss"],
                            "ms_loss": last_scalar["ms_loss"],
                            "log_aux_loss": last_scalar.get("log_aux_loss"),
                            "topk_peak_loss": last_scalar.get("topk_peak_loss"),
                            "peak_ratio_penalty": last_scalar.get("peak_ratio_penalty"),
                            "residual_loss": last_scalar["residual_loss"],
                            "sum_rule_loss": last_scalar["sum_rule_loss"],
                            "rel_model": last_rel_model,
                            "lr": float(opt.param_groups[0]["lr"]) if opt.param_groups else None,
                        },
                        components["train_metrics_jsonl"],
                    )

                if ctx.dist.is_main and step > 0 and step % ckpt_every == 0:
                    self._save_green_checkpoint(model_core, ckpt_mgr, step, components)

                if pbar is not None:
                    pbar.update(1)
                step += 1
                if step >= max_steps:
                    break

        if ctx.dist.is_main and components.get("last_ckpt") is None:
            self._save_green_checkpoint(model_core, ckpt_mgr, max(step, 1), components)
        if pbar is not None:
            pbar.close()

        wall = time.time() - t0
        steps_done = max(0, step - start_step)
        sps = float(steps_done) / max(wall, 1.0e-6)
        last_ckpt = components.get("last_ckpt") or ckpt_mgr.find_latest("latent_green_step_*.pt")
        return StageResult(
            stage=self.stage_name,
            success=True,
            step=ckpt_step_from_path(last_ckpt),
            metrics={
                "final_loss": last_scalar.get("loss"),
                "final_data_loss": last_scalar.get("data_loss"),
                "final_rel_model": last_rel_model,
                "wall_time_s": wall,
                "steps_per_sec": sps,
            },
            artifacts={
                "checkpoints_dir": cfg["paths"]["checkpoints"],
                "last_checkpoint": last_ckpt,
                "logs_dir": cfg["paths"]["logs"],
                "train_metrics_jsonl": components["train_metrics_jsonl"],
            },
            message="Latent Green training completed",
        )

    def _save_green_checkpoint(self, model_core: torch.nn.Module, ckpt_mgr: CheckpointManager, step: int, components: Dict[str, Any]) -> str:
        save_target = model_core._orig_mod if hasattr(model_core, "_orig_mod") else model_core
        ckpt_path = ckpt_mgr.save_state_dict("latent_green", step, save_target.state_dict(), torch_module=torch)
        components["last_ckpt"] = ckpt_path
        return ckpt_path
