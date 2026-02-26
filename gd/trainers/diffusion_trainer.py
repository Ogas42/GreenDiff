from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import deque
from typing import Any, Dict

import torch

from gd.core.checkpoints.manager import CheckpointManager
from gd.core.data.loader_factory import build_train_dataloader
from gd.core.logging.progress import get_tqdm
from gd.core.typing.types import ResumeState, StageResult
from gd.trainers.base import StageTrainer
from gd.trainers.diffusion_builder import (
    build_diffusion_model,
    build_diffusion_optimizer,
    build_ema_model_if_enabled,
    build_frozen_latent_green,
    build_frozen_vae,
    ckpt_step_from_path,
    load_latent_green_checkpoint,
    load_vae_checkpoint,
    prepare_diffusion_train_metrics_path,
    resume_diffusion_ema,
    resume_diffusion_model,
    save_diffusion_checkpoint,
    save_diffusion_ema_checkpoint,
)
from gd.trainers.diffusion_components import (
    append_diffusion_train_metric_jsonl,
    build_diffusion_scheduler,
    compute_total_diffusion_loss,
    diffusion_train_step,
    log_diffusion_train_status,
    prepare_latent_batch,
    sample_diffusion_training_target,
    update_ema_model,
)
from gd.utils.ldos_transform import force_linear_ldos_mode


class DiffusionTrainer(StageTrainer):
    stage_name = "diffusion"
    requires = ["vae", "green"]

    @staticmethod
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

    @classmethod
    def _validate_inverse_target_cfg(cls, cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> None:
        inverse_target = str(train_cfg.get("inverse_target", "v_only")).lower()
        allow_unclosed = cls._cfg_bool(train_cfg.get("allow_unclosed_inverse", False), default=False)
        structural_cfg = cfg.get("potential_sampler", {}).get("structural", {})
        structural_enabled = bool(structural_cfg.get("enabled", False)) if isinstance(structural_cfg, dict) else False
        if inverse_target == "v_only" and structural_enabled and not allow_unclosed:
            raise ValueError(
                "Diffusion inverse target is 'v_only' but potential_sampler.structural.enabled=true. "
                "This makes the inverse task unclosed because LDOS depends on V + structural defects. "
                "Set potential_sampler.structural.enabled=false for V-only diffusion, "
                "or set diffusion.training.allow_unclosed_inverse=true to override for experiments."
            )

    @staticmethod
    def _find_green_metrics_file(cfg: Dict[str, Any], gate_cfg: Dict[str, Any]) -> str | None:
        path_cfg = gate_cfg.get("green_metrics_path", "auto_latest")
        if isinstance(path_cfg, str) and path_cfg and path_cfg != "auto_latest":
            return path_cfg if os.path.isfile(path_cfg) else None
        workdir = cfg.get("paths", {}).get("workdir", "")
        runs_root = cfg.get("paths", {}).get("runs_root", "")
        candidates = []
        if workdir:
            candidates.append(os.path.join(workdir, "metrics", "green_eval_val.json"))
            candidates.extend(glob.glob(os.path.join(workdir, "metrics", "*green_eval*.json")))
        if runs_root and os.path.isdir(runs_root):
            candidates.extend(glob.glob(os.path.join(runs_root, "**", "green_eval_val.json"), recursive=True))
        existing = [p for p in candidates if os.path.isfile(p)]
        if not existing:
            return None
        existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return existing[0]

    @classmethod
    def _resolve_phys_gate_state(cls, cfg: Dict[str, Any], train_cfg: Dict[str, Any], *, is_main: bool) -> Dict[str, Any]:
        gate_cfg = train_cfg.get("phys_gate", {})
        if not isinstance(gate_cfg, dict):
            gate_cfg = {}
        enabled = cls._cfg_bool(gate_cfg.get("enabled", False), default=False)
        state: Dict[str, Any] = {
            "enabled": enabled,
            "passed": True,
            "reason": None,
            "metrics_path": None,
            "metrics": None,
        }
        if not enabled:
            return state

        allow_override = cls._cfg_bool(gate_cfg.get("allow_override", False), default=False)
        path = cls._find_green_metrics_file(cfg, gate_cfg)
        state["metrics_path"] = path
        if path is None:
            state["passed"] = bool(allow_override)
            state["reason"] = "green_metrics_missing" if not state["passed"] else "green_metrics_missing_override"
            return state
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            metrics = payload.get("metrics", payload if isinstance(payload, dict) else {})
            if not isinstance(metrics, dict):
                metrics = {}
            state["metrics"] = metrics
        except Exception as exc:
            state["passed"] = bool(allow_override)
            state["reason"] = f"green_metrics_read_error:{type(exc).__name__}"
            return state

        def _pick(*keys):
            for k in keys:
                v = state["metrics"].get(k) if isinstance(state["metrics"], dict) else None
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        return None
            return None

        rel = _pick("rel_l2", "rel_phys_mean", "rel_l2_mean")
        peak = _pick("peak_ratio_mean", "pred_obs_max_ratio_mean", "peak_ratio")
        mean_ratio = _pick("pred_obs_mean_ratio_mean", "mean_ratio_mean", "mean_ratio")
        state["metrics_compact"] = {"rel_l2": rel, "peak_ratio_mean": peak, "mean_ratio": mean_ratio}
        if rel is None or peak is None or mean_ratio is None:
            state["passed"] = bool(allow_override)
            state["reason"] = "green_metrics_missing_required_keys" if not state["passed"] else "override_missing_keys"
            return state

        rel_max = float(gate_cfg.get("rel_l2_max", 0.60))
        peak_max = float(gate_cfg.get("peak_ratio_max", 5.0))
        mean_min = float(gate_cfg.get("mean_ratio_min", 0.70))
        mean_max = float(gate_cfg.get("mean_ratio_max", 1.30))
        passed = (rel <= rel_max) and (peak <= peak_max) and (mean_min <= mean_ratio <= mean_max)
        if not passed and allow_override:
            passed = True
            state["reason"] = "override_thresholds_failed"
        elif not passed:
            state["reason"] = f"green_gate_fail(rel={rel:.3f},peak={peak:.3f},mean={mean_ratio:.3f})"
        else:
            state["reason"] = "green_gate_pass"
        state["passed"] = passed
        if is_main:
            print(
                f"[gd.diffusion_trainer] phys_gate enabled={enabled} passed={state['passed']} "
                f"reason={state['reason']} metrics_path={path}"
            )
        return state

    def build(self, ctx: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        force_linear_ldos_mode(cfg, verbose=ctx.dist.is_main, context="gd.diffusion_trainer")
        train_cfg = cfg["diffusion"]["training"]
        self._validate_inverse_target_cfg(cfg, train_cfg)
        phys_gate_state = self._resolve_phys_gate_state(cfg, train_cfg, is_main=ctx.dist.is_main)
        device = torch.device(ctx.dist.device)
        dataset, sampler, loader = build_train_dataloader(cfg, train_cfg, ctx.dist, split="train")
        ckpt_mgr = CheckpointManager(cfg["paths"]["runs_root"], cfg["paths"]["checkpoints"])

        vae = build_frozen_vae(cfg, device)
        latent_green = build_frozen_latent_green(cfg, device)
        model, model_core = build_diffusion_model(cfg, device, ctx.dist)
        opt = build_diffusion_optimizer(model, train_cfg)
        scheduler = build_diffusion_scheduler(opt, train_cfg)
        ema_model, ema_decay = build_ema_model_if_enabled(
            model_core=model_core, train_cfg=train_cfg, device=device, is_main=ctx.dist.is_main
        )
        train_metrics_jsonl = prepare_diffusion_train_metrics_path(cfg["paths"]["workdir"])
        images_dir = os.path.join(cfg["paths"]["workdir"], "images")
        os.makedirs(images_dir, exist_ok=True)
        return {
            "device": device,
            "train_cfg": train_cfg,
            "dataset": dataset,
            "sampler": sampler,
            "loader": loader,
            "vae": vae,
            "latent_green": latent_green,
            "model": model,
            "model_core": model_core,
            "optimizer": opt,
            "scheduler": scheduler,
            "ema_model": ema_model,
            "ema_decay": ema_decay,
            "ckpt_mgr": ckpt_mgr,
            "train_metrics_jsonl": train_metrics_jsonl,
            "images_dir": images_dir,
            "phys_gate_state": phys_gate_state,
            "last_ckpt": None,
            "last_ema_ckpt": None,
        }

    def resume(self, ctx: Any, cfg: Dict[str, Any], components: Dict[str, Any]) -> ResumeState:
        device = components["device"]
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        vae = components["vae"]
        latent_green = components["latent_green"]
        model_core = components["model_core"]
        ema_model = components.get("ema_model")

        load_vae_checkpoint(ckpt_mgr=ckpt_mgr, vae=vae, device=device, is_main=ctx.dist.is_main)
        load_latent_green_checkpoint(
            ckpt_mgr=ckpt_mgr,
            latent_green=latent_green,
            model_core=model_core,
            cfg=cfg,
            device=device,
            is_main=ctx.dist.is_main,
        )
        resume_state = resume_diffusion_model(ckpt_mgr=ckpt_mgr, model_core=model_core, device=device, is_main=ctx.dist.is_main)
        ema_path = resume_diffusion_ema(
            ema_model=ema_model,
            resume_state=resume_state,
            ckpt_mgr=ckpt_mgr,
            device=device,
            is_main=ctx.dist.is_main,
        )
        components["last_ckpt"] = resume_state.checkpoint_path
        components["last_ema_ckpt"] = ema_path
        return resume_state

    def run(self, ctx: Any, cfg: Dict[str, Any], components: Dict[str, Any], resume_state: ResumeState) -> StageResult:
        t0 = time.time()
        device = components["device"]
        train_cfg = components["train_cfg"]
        loader = components["loader"]
        sampler = components["sampler"]
        vae = components["vae"]
        latent_green = components["latent_green"]
        model = components["model"]
        model_core = components["model_core"]
        opt = components["optimizer"]
        scheduler = components["scheduler"]
        ema_model = components.get("ema_model")
        ema_decay = components.get("ema_decay")
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        amp = ctx.amp
        phys_gate_state = components.get("phys_gate_state")

        max_steps = int(train_cfg["max_steps"])
        log_every = int(train_cfg["log_every"])
        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        ckpt_every = int(train_cfg.get("ckpt_every", 2000))
        show_progress_bar = self._cfg_bool(train_cfg.get("show_progress_bar", False), default=False)
        if not sys.stderr.isatty():
            # Non-interactive log collectors often print a new line for every tqdm refresh.
            show_progress_bar = False
        if ctx.dist.is_main:
            print(
                f"[gd.diffusion_trainer] show_progress_bar={show_progress_bar} "
                f"(cfg={train_cfg.get('show_progress_bar', None)!r}, stderr_tty={sys.stderr.isatty()})"
            )
        step = int(resume_state.step)
        start_step = step

        data_cfg = cfg.get("data", {})
        prediction_type = str(train_cfg.get("prediction_type", "eps"))

        tqdm = get_tqdm()
        pbar = (
            tqdm(total=max_steps, initial=step, desc="Training Diffusion", dynamic_ncols=True)
            if (ctx.dist.is_main and show_progress_bar)
            else None
        )
        smooth_window = int(train_cfg.get("log_smooth_window", 50))
        histories = {
            "loss": deque(maxlen=max(1, smooth_window)),
            "base": deque(maxlen=max(1, smooth_window)),
            "x0": deque(maxlen=max(1, smooth_window)),
            "phys": deque(maxlen=max(1, smooth_window)),
            "cons": deque(maxlen=max(1, smooth_window)),
            "phys_coeff": deque(maxlen=max(1, smooth_window)),
        }
        last_log_time = time.perf_counter()
        last_loss_pack: Dict[str, Any] = {}

        while step < max_steps:
            if sampler is not None:
                epoch = step // max(1, len(loader))
                sampler.set_epoch(epoch)
            for batch in loader:
                V = batch["V"].to(device, non_blocking=True)
                if V.dim() == 3:
                    V = V.unsqueeze(1)
                g_obs = batch["g_obs"].to(device, non_blocking=True)

                with torch.no_grad():
                    z, latent_meta = prepare_latent_batch(vae=vae, V=V, train_cfg=train_cfg)
                sample = sample_diffusion_training_target(model_core=model_core, z=z, prediction_type=prediction_type)
                loss_pack = compute_total_diffusion_loss(
                    cfg=cfg,
                    model=model,
                    model_core=model_core,
                    latent_green=latent_green,
                    z=z,
                    g_obs=g_obs,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    step=step,
                    amp=amp,
                    sample=sample,
                    phys_gate_state=phys_gate_state,
                )
                diffusion_train_step(
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    loss=loss_pack["loss"],
                    grad_clip=grad_clip,
                    amp=amp,
                )
                if ctx.dist.is_main:
                    update_ema_model(ema_model=ema_model, model_core=model_core, ema_decay=ema_decay)

                if ctx.dist.is_main and step % log_every == 0:
                    last_log_time = log_diffusion_train_status(
                        step=step,
                        train_cfg=train_cfg,
                        opt=opt,
                        z=z,
                        loss_pack=loss_pack,
                        histories=histories,
                        pbar=pbar,
                        start_or_last_log_time=last_log_time,
                    )
                    append_diffusion_train_metric_jsonl(
                        path=components["train_metrics_jsonl"],
                        step=step,
                        opt=opt,
                        loss_pack=loss_pack,
                    )
                last_loss_pack = loss_pack

                if ctx.dist.is_main and step > 0 and step % ckpt_every == 0:
                    self._save_checkpoints(step=step, components=components)

                if pbar is not None:
                    pbar.update(1)
                step += 1
                if step >= max_steps:
                    break

        if ctx.dist.is_main and components.get("last_ckpt") is None:
            self._save_checkpoints(step=max(step, 1), components=components)
        if pbar is not None:
            pbar.close()

        wall = time.time() - t0
        steps_done = max(0, step - start_step)
        sps = float(steps_done) / max(wall, 1.0e-6)
        last_ckpt = components.get("last_ckpt") or ckpt_mgr.find_latest("diffusion_step_*.pt")
        last_ema_ckpt = components.get("last_ema_ckpt")
        return StageResult(
            stage=self.stage_name,
            success=True,
            step=ckpt_step_from_path(last_ckpt),
            metrics={
                "final_loss": self._loss_scalar(last_loss_pack, "loss"),
                "final_diffusion_loss": self._loss_scalar(last_loss_pack, "base_loss"),
                "final_x0_loss": self._loss_scalar(last_loss_pack, "x0_loss"),
                "final_phys_loss": self._loss_scalar(last_loss_pack, "phys_loss"),
                "final_consistency_loss": self._loss_scalar(last_loss_pack, "consistency_loss"),
                "wall_time_s": wall,
                "steps_per_sec": sps,
            },
            artifacts={
                "checkpoints_dir": cfg["paths"]["checkpoints"],
                "last_checkpoint": last_ckpt,
                "last_ema_checkpoint": last_ema_ckpt,
                "train_metrics_jsonl": components["train_metrics_jsonl"],
                "images_dir": components["images_dir"],
                "logs_dir": cfg["paths"]["logs"],
            },
            message=(
                "Diffusion training completed "
                f"(ema={'on' if ema_model is not None else 'off'}, "
                f"phys_loss={float(train_cfg.get('phys_loss_weight', 0.0)):.3g})"
            ),
        )

    def _save_checkpoints(self, *, step: int, components: Dict[str, Any]) -> None:
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        model_core = components["model_core"]
        ema_model = components.get("ema_model")
        components["last_ckpt"] = save_diffusion_checkpoint(
            ckpt_mgr=ckpt_mgr,
            model_core=model_core,
            step=step,
            torch_module=torch,
        )
        components["last_ema_ckpt"] = save_diffusion_ema_checkpoint(
            ema_model=ema_model,
            ckpt_mgr=ckpt_mgr,
            step=step,
            torch_module=torch,
        )

    @staticmethod
    def _loss_scalar(loss_pack: Dict[str, Any], key: str) -> float | None:
        if not loss_pack or key not in loss_pack:
            return None
        val = loss_pack.get(key)
        if val is None:
            return None
        if torch.is_tensor(val):
            return float(val.detach().item())
        try:
            return float(val)
        except Exception:
            return None
