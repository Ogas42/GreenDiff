from __future__ import annotations

import os
import re
import time
from typing import Any, Dict

import torch
import torch.optim as optim

from gd.core.checkpoints.manager import CheckpointManager, normalize_state_dict_keys
from gd.core.data.loader_factory import build_train_dataloader
from gd.core.logging.progress import get_tqdm
from gd.core.typing.types import ResumeState, StageResult
from gd.trainers.base import StageTrainer
from gd.utils.ldos_transform import force_linear_ldos_mode


def _ckpt_step(path: str) -> int:
    m = re.search(r"_step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else 0


class VAETrainer(StageTrainer):
    stage_name = "vae"
    requires = []

    def build(self, ctx: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
        force_linear_ldos_mode(cfg, verbose=ctx.dist.is_main, context="gd.vae_trainer")
        train_cfg = cfg["vae"]["training"]
        dataset, sampler, loader = build_train_dataloader(cfg, train_cfg, ctx.dist, split="train")

        from gd.models.vae import VAE

        device = torch.device(ctx.dist.device)
        model = VAE(cfg).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = bool(cfg["project"].get("cudnn_benchmark", False))
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

        if cfg["project"].get("compile", False) and hasattr(torch, "compile") and not ctx.dist.is_distributed:
            if ctx.dist.is_main:
                print("Enabling torch.compile (gd vae trainer)...")
            model = torch.compile(model)

        if ctx.dist.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ctx.dist.local_rank] if str(device).startswith("cuda") else None,
                output_device=ctx.dist.local_rank if str(device).startswith("cuda") else None,
            )
        model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        opt = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
        ckpt_mgr = CheckpointManager(cfg["paths"]["runs_root"], cfg["paths"]["checkpoints"])
        return {
            "device": device,
            "train_cfg": train_cfg,
            "dataset": dataset,
            "sampler": sampler,
            "loader": loader,
            "model": model,
            "model_core": model_core,
            "optimizer": opt,
            "ckpt_mgr": ckpt_mgr,
            "last_ckpt": None,
        }

    def resume(self, ctx: Any, cfg: Dict[str, Any], components: Dict[str, Any]) -> ResumeState:
        ckpt_path = components["ckpt_mgr"].find_latest("vae_step_*.pt")
        if not ckpt_path:
            if ctx.dist.is_main:
                print("No VAE checkpoint found. Starting from scratch.")
            return ResumeState(step=0, checkpoint_path=None)
        if ctx.dist.is_main:
            print(f"Resuming VAE from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=components["device"], weights_only=True)
        state_dict = normalize_state_dict_keys(state_dict)
        try:
            load_target = components["model_core"]._orig_mod if hasattr(components["model_core"], "_orig_mod") else components["model_core"]
            load_target.load_state_dict(state_dict)
            step = _ckpt_step(ckpt_path)
            return ResumeState(step=step, checkpoint_path=ckpt_path)
        except RuntimeError as e:
            if ctx.dist.is_main:
                print(f"Warning: failed to resume VAE checkpoint ({e}); restarting from scratch.")
            return ResumeState(step=0, checkpoint_path=None, meta={"resume_error": str(e)})

    def run(self, ctx: Any, cfg: Dict[str, Any], components: Dict[str, Any], resume_state: ResumeState) -> StageResult:
        device = components["device"]
        loader = components["loader"]
        sampler = components["sampler"]
        model = components["model"]
        model_core = components["model_core"]
        opt = components["optimizer"]
        train_cfg = components["train_cfg"]
        ckpt_mgr: CheckpointManager = components["ckpt_mgr"]
        amp = ctx.amp

        max_steps = int(train_cfg["max_steps"])
        log_every = int(train_cfg["log_every"])
        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        ckpt_every = int(train_cfg.get("ckpt_every", 2000))
        step = int(resume_state.step)

        tqdm = get_tqdm()
        pbar = tqdm(total=max_steps, initial=step, desc="Training VAE") if ctx.dist.is_main else None
        t0 = time.time()
        last_losses: Dict[str, float] = {}

        while step < max_steps:
            if sampler is not None:
                epoch = step // max(1, len(loader))
                sampler.set_epoch(epoch)
            for batch in loader:
                V = batch["V"].to(device, non_blocking=True)
                if V.dim() == 3:
                    V = V.unsqueeze(1)
                if device.type == "cuda":
                    V = V.to(memory_format=torch.channels_last)

                with torch.amp.autocast(
                    "cuda" if device.type == "cuda" else "cpu",
                    enabled=bool(amp.use_amp),
                    dtype=amp.amp_dtype,
                ):
                    V_hat, mu, logvar = model(V)
                    losses = model_core.loss(V, V_hat, mu, logvar)
                    loss = losses["loss"]

                opt.zero_grad(set_to_none=True)
                if amp.use_scaler and amp.scaler is not None:
                    amp.scaler.scale(loss).backward()
                    if grad_clip > 0:
                        amp.scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    amp.scaler.step(opt)
                    amp.scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()

                last_losses = {k: float(v.detach().item()) for k, v in losses.items()}
                if ctx.dist.is_main and step % log_every == 0 and pbar is not None:
                    pbar.set_postfix(
                        {
                            "loss": f"{last_losses.get('loss', 0.0):.6f}",
                            "recon": f"{last_losses.get('recon_loss', 0.0):.6f}",
                            "kl": f"{last_losses.get('kl_loss', 0.0):.6f}",
                        }
                    )

                if ctx.dist.is_main and step > 0 and step % ckpt_every == 0:
                    save_target = model_core._orig_mod if hasattr(model_core, "_orig_mod") else model_core
                    ckpt_path = ckpt_mgr.save_state_dict("vae", step, save_target.state_dict(), torch_module=torch)
                    components["last_ckpt"] = ckpt_path

                if pbar is not None:
                    pbar.update(1)
                step += 1
                if step >= max_steps:
                    break

        if ctx.dist.is_main and components.get("last_ckpt") is None:
            save_target = model_core._orig_mod if hasattr(model_core, "_orig_mod") else model_core
            components["last_ckpt"] = ckpt_mgr.save_state_dict("vae", max(step, 1), save_target.state_dict(), torch_module=torch)
        if pbar is not None:
            pbar.close()
        wall = time.time() - t0
        msg = "VAE training completed"
        return StageResult(
            stage=self.stage_name,
            success=True,
            step=step,
            metrics={
                "final_loss": last_losses.get("loss"),
                "final_recon_loss": last_losses.get("recon_loss"),
                "final_kl_loss": last_losses.get("kl_loss"),
                "wall_time_s": wall,
            },
            artifacts={
                "checkpoints_dir": cfg["paths"]["checkpoints"],
                "last_checkpoint": components.get("last_ckpt"),
                "logs_dir": cfg["paths"]["logs"],
            },
            message=msg,
        )
