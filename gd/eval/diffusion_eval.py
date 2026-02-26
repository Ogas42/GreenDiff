from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from gd.core.checkpoints.manager import CheckpointManager
from gd.core.logging.progress import get_tqdm
from gd.core.logging.results import (
    checkpoint_fingerprint,
    config_fingerprint,
    hardware_info,
    save_eval_result_json,
    utc_timestamp,
)
from gd.data.dataset import GFDataset
from gd.trainers.diffusion_builder import (
    build_diffusion_model,
    build_frozen_latent_green,
    build_frozen_vae,
    load_latent_green_checkpoint,
    load_vae_checkpoint,
    resume_diffusion_ema,
    resume_diffusion_model,
)
from gd.trainers.diffusion_components import prepare_latent_batch
from gd.trainers.diffusion_validation import (
    build_teacher_sampler_for_eval,
    compute_diffusion_eval_metrics,
    render_diffusion_comparison_grid,
    sample_diffusion_predictions,
    summarize_metric_lists,
)
from gd.utils.config_utils import resolve_config_paths
from gd.utils.ldos_transform import force_linear_ldos_mode


def _prepare_config(config: Dict[str, Any], ckpt_dir: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    force_linear_ldos_mode(cfg, verbose=True, context="gd.eval.diffusion_eval")
    if ckpt_dir:
        run_dir = os.path.dirname(ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        cfg["paths"]["checkpoints"] = ckpt_dir
        cfg["paths"]["workdir"] = run_dir
        cfg["paths"]["runs_root"] = runs_root
        cfg = resolve_config_paths(cfg, cfg.get("paths", {}).get("config_path"))
    return cfg


def _eval_dist_stub() -> Any:
    return SimpleNamespace(is_distributed=False, local_rank=0)


def _load_eval_models(
    cfg: Dict[str, Any],
    *,
    device: torch.device,
    ckpt_dir: Optional[str],
    use_ema: bool,
) -> Dict[str, Any]:
    current_ckpt_dir = ckpt_dir or cfg["paths"]["checkpoints"]
    ckpt_mgr = CheckpointManager(cfg["paths"]["runs_root"], current_ckpt_dir)
    vae = build_frozen_vae(cfg, device)
    latent_green = build_frozen_latent_green(cfg, device)
    model, model_core = build_diffusion_model(cfg, device, _eval_dist_stub())

    vae_ckpt = load_vae_checkpoint(ckpt_mgr=ckpt_mgr, vae=vae, device=device, is_main=True)
    lg_ckpt = load_latent_green_checkpoint(
        ckpt_mgr=ckpt_mgr,
        latent_green=latent_green,
        model_core=model_core,
        cfg=cfg,
        device=device,
        is_main=True,
    )
    if not lg_ckpt:
        raise FileNotFoundError("Latent Green checkpoint is required for diffusion evaluation.")
    resume_state = resume_diffusion_model(ckpt_mgr=ckpt_mgr, model_core=model_core, device=device, is_main=True)
    if not resume_state.checkpoint_path:
        raise FileNotFoundError("Diffusion checkpoint is required for diffusion evaluation.")

    eval_model = model_core
    ema_model = None
    ema_ckpt = None
    if use_ema:
        ema_model = copy.deepcopy(model_core).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_ckpt = resume_diffusion_ema(
            ema_model=ema_model,
            resume_state=resume_state,
            ckpt_mgr=ckpt_mgr,
            device=device,
            is_main=True,
        )
        if ema_ckpt:
            eval_model = ema_model

    return {
        "ckpt_mgr": ckpt_mgr,
        "vae": vae.eval(),
        "latent_green": latent_green.eval(),
        "model": model,
        "model_core": model_core.eval(),
        "eval_model": eval_model.eval(),
        "vae_ckpt": vae_ckpt,
        "lg_ckpt": lg_ckpt,
        "diff_ckpt": resume_state.checkpoint_path,
        "ema_ckpt": ema_ckpt,
    }


def run(
    config: Dict[str, Any],
    runtime_ctx: Any = None,
    ckpt_dir: Optional[str] = None,
    split: str = "val",
    max_batches: Optional[int] = None,
    save_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    dataset_suite: str = "synthetic_main_v1",
    variant: Optional[str] = None,
    quiet: bool = False,
    vis_n: Optional[int] = None,
    use_ema: bool = True,
    batch_size: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    cfg = _prepare_config(config, ckpt_dir)
    device = torch.device(runtime_ctx.dist.device if runtime_ctx is not None else cfg["project"]["device"])
    if max_batches is None:
        max_batches = int(kwargs.get("num_batches", 20))
    if batch_size is None:
        batch_size = int(kwargs.get("batch_size", cfg["diffusion"]["training"].get("batch_size", 4)))
    if vis_n is None:
        vis_n = int(kwargs.get("vis_n", cfg["diffusion"]["training"].get("vis_fixed_n", 0) or 0))

    models = _load_eval_models(cfg, device=device, ckpt_dir=ckpt_dir, use_ema=bool(use_ema))
    vae = models["vae"]
    latent_green = models["latent_green"]
    eval_model = models["eval_model"]

    dataset = GFDataset(cfg, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sampler = build_teacher_sampler_for_eval(cfg, diffusion_model=eval_model, vae=vae, latent_green=latent_green)
    metric_lists: Dict[str, list[float]] = {"mse": [], "mae": [], "rel_l2": [], "psd_error": []}
    vis_image_paths: list[str] = []
    run_dir = cfg["paths"]["workdir"]
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    tqdm = get_tqdm()
    total = min(len(loader), max_batches if max_batches is not None else len(loader))
    pbar = tqdm(loader, total=total, disable=quiet)
    eval_batches = 0

    for idx, batch in enumerate(pbar):
        if max_batches is not None and idx >= max_batches:
            break
        V_true = batch["V"].to(device)
        if V_true.dim() == 3:
            V_true = V_true.unsqueeze(1)
        g_obs = batch["g_obs"].to(device)
        defect_meta = batch.get("defect_meta") if isinstance(batch, dict) else None

        with torch.no_grad():
            _z_tmp, latent_meta = prepare_latent_batch(vae=vae, V=V_true, train_cfg=cfg["diffusion"]["training"])
            V_pred = sample_diffusion_predictions(
                sampler=sampler,
                g_obs=g_obs,
                latent_unscale_factor=latent_meta.get("unscale_factor"),
            )
        batch_metrics = compute_diffusion_eval_metrics(V_pred, V_true)
        for k, v in batch_metrics.items():
            metric_lists[k].append(float(v))
        eval_batches += 1

        if vis_n and idx == 0:
            try:
                save_path = os.path.join(images_dir, f"diffusion_eval_{split}.png")
                defect_meta_vis = None
                if isinstance(defect_meta, dict):
                    defect_meta_vis = {
                        k: (v[:vis_n] if torch.is_tensor(v) and v.dim() >= 1 else v)
                        for k, v in defect_meta.items()
                    }
                vis_path = render_diffusion_comparison_grid(
                    V_true=V_true[:vis_n],
                    V_pred=V_pred[:vis_n],
                    g_obs=g_obs[:vis_n],
                    defect_meta=defect_meta_vis,
                    save_path=save_path,
                    title_prefix="Diffusion Eval",
                )
                vis_image_paths.append(vis_path)
            except Exception as exc:
                if not quiet:
                    print(f"[gd.eval.diffusion_eval] visualization failed: {exc}")

        if not quiet and hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"rel": f"{metric_lists['rel_l2'][-1]:.4f}", "mse": f"{metric_lists['mse'][-1]:.4f}"})

    summary_metrics = summarize_metric_lists(metric_lists)
    ckpt_path = models["ema_ckpt"] if (bool(use_ema) and models.get("ema_ckpt")) else models["diff_ckpt"]
    run_id = os.path.basename(run_dir)
    result = {
        "task": "diffusion_eval",
        "dataset_suite": dataset_suite,
        "variant": variant,
        "split": split,
        "seed": int(cfg["project"]["seed"]),
        "run_id": run_id,
        "metrics": summary_metrics,
        "artifacts": {"vis_image_paths": vis_image_paths},
        "config_hash": config_fingerprint(cfg),
        "checkpoint_tag": os.path.basename(ckpt_path) if ckpt_path else None,
        "checkpoint_hash": checkpoint_fingerprint(ckpt_path),
        "timestamp": utc_timestamp(),
        "hardware": hardware_info(),
        "meta": {
            "variant": variant,
            "use_ema": bool(use_ema),
            "eval_batches": eval_batches,
            "vis_n": int(vis_n or 0),
            "batch_size": int(batch_size),
            "vae_checkpoint": os.path.basename(models["vae_ckpt"]) if models.get("vae_ckpt") else None,
            "latent_green_checkpoint": os.path.basename(models["lg_ckpt"]) if models.get("lg_ckpt") else None,
            "diffusion_checkpoint": os.path.basename(models["diff_ckpt"]) if models.get("diff_ckpt") else None,
            "ema_checkpoint": os.path.basename(models["ema_ckpt"]) if models.get("ema_ckpt") else None,
        },
    }

    if save_json is None and output_dir:
        save_json = os.path.join(output_dir, f"{run_id}_diffusion_eval_{split}.json")
    if save_json:
        result["artifacts"]["result_json"] = save_eval_result_json(result, save_json)

    return result
