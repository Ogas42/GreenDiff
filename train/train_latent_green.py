import os
import glob
import math
import time
from typing import Dict, Any
from collections import deque
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from gd.data.dataset import GFDataset
from gd.models.vae import VAE
from gd.models.latent_green import LatentGreen
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_linear_from_obs, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import (
    aggregate_sublattice_ldos,
    flatten_sub_for_energy_ops,
    g_obs_to_canonical_view,
    is_sublattice_resolved,
)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _to_device_tree(x, device, non_blocking: bool = True):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_device_tree(v, device, non_blocking=non_blocking) for k, v in x.items()}
    return x


def _green_physics_weights(model_cfg: Dict[str, Any]) -> Dict[str, float]:
    phy = model_cfg.get("physics_losses", {}) if isinstance(model_cfg.get("physics_losses", {}), dict) else {}
    return {
        "data_weight": float(phy.get("data_weight", 1.0)),
        "residual_weight": float(phy.get("residual_weight", model_cfg.get("residual_loss_weight", 0.0))),
        "sum_rule_weight": float(phy.get("sum_rule_weight", 0.0)),
        "nonneg_weight": float(phy.get("nonneg_weight", 0.0)),
    }

def train_latent_green(config: Dict[str, Any]):
    """
    Main training loop for the Latent Green's Function model.
    """
    device = torch.device(config["project"]["device"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    if is_distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    is_main = (not is_distributed) or rank == 0
    force_linear_ldos_mode(config, verbose=is_main, context="train_latent_green")
    data_cfg = config["latent_green"]["training"]
    precision = config["project"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_scaler = use_amp and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    common_data_cfg = config.get("data", {})
    sublattice_resolved = bool(is_sublattice_resolved(common_data_cfg))
    dataset = GFDataset(config, split="train")
    shuffle = common_data_cfg.get("shuffle")
    if shuffle is None:
        shuffle = not dataset.use_shards
    num_workers = common_data_cfg.get("num_workers", 0)
    if dataset.use_shards:
        shard_workers = common_data_cfg.get("shard_workers")
        if shard_workers is None:
            num_workers = min(num_workers, 4)
        else:
            num_workers = shard_workers
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if is_distributed else None
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        pin_memory=common_data_cfg["pin_memory"],
        sampler=sampler,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = common_data_cfg["persistent_workers"]
        loader_kwargs["prefetch_factor"] = common_data_cfg.get("prefetch_factor", 2)
    loader = DataLoader(**loader_kwargs)
    debug_fixed_batch = bool(data_cfg.get("debug_fixed_batch", False))
    fixed_batch = None
    if debug_fixed_batch:
        fixed_batch = next(iter(loader))
        if is_main:
            print("Debug: using fixed batch for training loop.")

    vae = VAE(config).to(device)
    
    # Determine Checkpoint Directory
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    if is_main:
        print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    # Helper to find latest checkpoint for a pattern
    def find_latest_ckpt(pattern):
        # 1. Try current run
        ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            return ckpts[-1]
        
        # 2. Try latest run containing this pattern
        latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
        if latest_dir:
            ckpts = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
            if ckpts:
                if is_main:
                    print(f"Found {pattern} in latest run: {latest_dir}")
                return ckpts[-1]
        return None

    def normalize_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_k = k[len('_orig_mod.'):]
            elif k.startswith('module.'):
                new_k = k[len('module.'):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    # Load VAE weights
    vae_ckpt = find_latest_ckpt("vae_step_*.pt")
    if vae_ckpt:
        if is_main:
            print(f"Loading VAE from {vae_ckpt}")
        vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        if is_main:
            print("Warning: No VAE checkpoint found! Training on random latents.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae_core = vae

    model = LatentGreen(config).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    # Resume Latent Green if checkpoint exists
    lg_ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, "latent_green_step_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(lg_ckpts) > 0:
        if is_main:
            print(f"Resuming Latent Green from {lg_ckpts[-1]}")
        try:
            model_core.load_state_dict(normalize_state_dict(torch.load(lg_ckpts[-1], map_location=device, weights_only=True)))
        except RuntimeError as e:
            raise RuntimeError(
                "Latent Green checkpoint is incompatible with the current LDOS channel layout "
                "(Phase-1 graphene A/B sublattice-resolved LDOS upgrades channels from K to 2K). "
                "Re-train the Latent Green stage with the new cache/schema."
            ) from e
        step = int(lg_ckpts[-1].split("_")[-1].split(".")[0])
        if is_main:
            print(f"Resuming from step {step}")
    else:
        if is_main:
            print("Starting Latent Green training from scratch...")
        step = 0

    opt = optim.AdamW(model.parameters(), lr=data_cfg["lr"], weight_decay=data_cfg["weight_decay"])
    max_steps = data_cfg["max_steps"]
    log_every = data_cfg["log_every"]
    grad_clip = data_cfg["grad_clip"]
    noisy_cfg = config["latent_green"]["noisy_latent_training"]
    model_cfg = config["latent_green"]["model"]
    phy_loss_cfg = model_cfg.get("physics_losses", {}) if isinstance(model_cfg.get("physics_losses", {}), dict) else {}
    need_residual = float(phy_loss_cfg.get("residual_weight", model_cfg.get("residual_loss_weight", 0.0))) > 0.0
    sched_cfg = data_cfg.get("lr_schedule", {})
    scheduler = None
    if sched_cfg.get("enabled"):
        warmup_steps = int(sched_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
        warmup_min_ratio = float(sched_cfg.get("warmup_min_ratio", min_lr_ratio))
        def lr_lambda(step_idx: int):
            if warmup_steps > 0 and step_idx < warmup_steps:
                ramp = step_idx / float(max(1, warmup_steps))
                return max(warmup_min_ratio, ramp)
            if max_steps <= warmup_steps:
                return 1.0
            progress = (step_idx - warmup_steps) / float(max_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    pbar = tqdm(total=max_steps, initial=step, desc="Training Latent Green", dynamic_ncols=True) if is_main else None
    smooth_window = int(data_cfg.get("log_smooth_window", 50))
    loss_hist = deque(maxlen=max(1, smooth_window))
    data_hist = deque(maxlen=max(1, smooth_window))
    rel_hist = deque(maxlen=max(1, smooth_window))
    last_log_time = time.perf_counter()
    while step < max_steps:
        if sampler is not None:
            epoch = step // max(1, len(loader))
            sampler.set_epoch(epoch)
        batch_iter = (fixed_batch,) if fixed_batch is not None else loader
        for batch in batch_iter:
            V = batch["V"].to(device, non_blocking=True).unsqueeze(1)
            g_obs = batch["g_obs"].to(device, non_blocking=True)
            physics_meta = _to_device_tree(batch.get("physics_meta"), device) if isinstance(batch, dict) else None
            defect_meta = _to_device_tree(batch.get("defect_meta"), device) if isinstance(batch, dict) else None
            with torch.no_grad():
                z, _, _ = vae_core.encode(V)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if noisy_cfg["enabled"]:
                    t = torch.randint(0, noisy_cfg["T"], (z.shape[0],), device=device)
                    z_t, _, _ = model_core.add_noise(z, t)
                    clean_prob = float(noisy_cfg.get("clean_prob", 0.0))
                    if clean_prob > 0.0:
                        clean_mask = torch.rand(z.shape[0], device=device) < clean_prob
                        if clean_mask.any():
                            t = t.clone()
                            t[clean_mask] = 0
                            z_t = z_t.clone()
                            z_t[clean_mask] = z[clean_mask]
                    if need_residual:
                        g_pred, psi_real, psi_imag, src = model(
                            z_t, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True
                        )
                    else:
                        g_pred = model(z_t, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=False)
                        psi_real = psi_imag = src = None
                else:
                    z_t = z
                    t = torch.zeros((z.shape[0],), dtype=torch.long, device=device)
                    if need_residual:
                        g_pred, psi_real, psi_imag, src = model(
                            z_t, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=True
                        )
                    else:
                        g_pred = model(z_t, t, physics_meta=physics_meta, defect_meta=defect_meta, return_psi=False)
                        psi_real = psi_imag = src = None
                if need_residual:
                    residual_loss = model_core.residual_loss(
                        psi_real, psi_imag, src, V, physics_meta=physics_meta, defect_meta=defect_meta
                    )
                else:
                    residual_loss = torch.zeros((), device=g_pred.device, dtype=g_pred.dtype)
                losses = model_core.loss(g_pred, g_obs, residual_loss, physics_meta=physics_meta)
            aux_warmup_steps = int(model_cfg.get("aux_warmup_steps", 0))
            if aux_warmup_steps > 0:
                aux_scale = min(1.0, step / float(aux_warmup_steps))
            else:
                aux_scale = 1.0
            fft_weight = model_cfg.get("fft_loss_weight", 0.0) if model_cfg.get("use_fft_loss", False) else 0.0
            stats_weight = model_cfg.get("stats_loss_weight", 0.0)
            psd_weight = model_cfg.get("psd_loss_weight", 0.0)
            linear_scale_weight = model_cfg.get("linear_scale_loss_weight", 0.0)
            ms_weight = model_cfg.get("multiscale_loss_weight", 0.0)
            phy_w = _green_physics_weights(model_cfg)
            total_loss = phy_w["data_weight"] * losses["data_loss"]
            total_loss = total_loss + aux_scale * fft_weight * losses["fft_loss"]
            total_loss = total_loss + aux_scale * psd_weight * losses.get("psd_loss", torch.zeros_like(losses["data_loss"]))
            total_loss = total_loss + aux_scale * stats_weight * losses["stats_loss"]
            total_loss = total_loss + linear_scale_weight * losses["linear_scale_loss"]
            total_loss = total_loss + aux_scale * ms_weight * losses["ms_loss"]
            total_loss = total_loss + phy_w["residual_weight"] * losses["residual_loss"]
            total_loss = total_loss + phy_w["sum_rule_weight"] * losses.get("sum_rule_loss", torch.zeros_like(losses["data_loss"]))
            total_loss = total_loss + phy_w["nonneg_weight"] * losses.get("nonneg_loss", torch.zeros_like(losses["data_loss"]))
            losses["loss"] = total_loss
            opt.zero_grad(set_to_none=True)
            optimizer_stepped = False
            if use_scaler:
                scaler.scale(losses["loss"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                optimizer_stepped = True
            else:
                losses["loss"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()

            if is_main and step % log_every == 0:
                now = time.perf_counter()
                elapsed = max(1.0e-6, now - last_log_time)
                it_s = float(log_every) / elapsed
                samples_s = it_s * float(data_cfg["batch_size"]) * float(world_size)
                last_log_time = now
                with torch.no_grad():
                    ldos_cfg = common_data_cfg.get("ldos_transform", {})
                    log_cfg = ldos_cfg.get("log", {})
                    log_enabled = ldos_cfg.get("enabled", False) and log_cfg.get("enabled", False)
                    model_cfg = config["latent_green"]["model"]
                    loss_type = model_cfg.get("loss_type", "mse")
                    huber_beta = float(model_cfg.get("huber_beta", 0.1))
                    use_per_energy_affine = bool(model_cfg.get("per_energy_affine", False))
                    align_cfg = model_cfg.get("energy_align", {})
                    align_enabled = bool(align_cfg.get("enabled", False))
                    align_max_shift = int(align_cfg.get("max_shift", 0))
                    log_cosh_eps = float(model_cfg.get("log_cosh_eps", 1.0e-6))
                    if sublattice_resolved:
                        g_pred_obs_flat = flatten_sub_for_energy_ops(
                            g_obs_to_canonical_view(ldos_obs_from_linear(g_pred, common_data_cfg), common_data_cfg)
                        )
                        g_obs_obs_flat = flatten_sub_for_energy_ops(g_obs)
                        g_pred_lin_flat = flatten_sub_for_energy_ops(g_obs_to_canonical_view(g_pred, common_data_cfg))
                        g_obs_lin_flat = flatten_sub_for_energy_ops(ldos_linear_from_obs(g_obs, common_data_cfg))
                    else:
                        g_pred_obs_flat = ldos_obs_from_linear(g_pred, common_data_cfg)
                        g_obs_obs_flat = g_obs
                        g_pred_lin_flat = g_pred
                        g_obs_lin_flat = ldos_linear_from_obs(g_obs, common_data_cfg)

                    if log_enabled:
                        g_pred_log = g_pred_obs_flat
                        g_obs_log = g_obs_obs_flat
                        if use_per_energy_affine:
                            g_pred_log = per_energy_affine(g_pred_log, g_obs_log)
                        g_pred_log, _ = align_pred(
                            g_pred_log,
                            g_obs_log,
                            enabled=align_enabled,
                            max_shift=align_max_shift,
                            loss_type=loss_type,
                            huber_beta=huber_beta,
                            log_cosh_eps=log_cosh_eps,
                        )
                        rel_l2 = torch.norm(g_pred_log - g_obs_log) / torch.norm(g_obs_log).clamp_min(1.0e-6)
                        pred_min = g_pred.min().item()
                        pred_max = g_pred.max().item()
                        pred_min_log = g_pred_log.min().item()
                        pred_max_log = g_pred_log.max().item()
                        pred_mean_log = g_pred_log.mean().item()
                        pred_std_log = g_pred_log.std().item()
                        obs_min_log = g_obs_log.min().item()
                        obs_max_log = g_obs_log.max().item()
                        obs_mean_log = g_obs_log.mean().item()
                        obs_std_log = g_obs_log.std().item()
                        g_obs_lin = g_obs_lin_flat
                        obs_min = g_obs_lin.min().item()
                        obs_max = g_obs_lin.max().item()
                    else:
                        g_pred_obs = g_pred_obs_flat
                        g_obs_obs = g_obs_obs_flat
                        if use_per_energy_affine:
                            g_pred_obs = per_energy_affine(g_pred_obs, g_obs_obs)
                        g_pred_obs, _ = align_pred(
                            g_pred_obs,
                            g_obs_obs,
                            enabled=align_enabled,
                            max_shift=align_max_shift,
                            loss_type=loss_type,
                            huber_beta=huber_beta,
                            log_cosh_eps=log_cosh_eps,
                        )
                        rel_l2 = torch.norm(g_pred_obs - g_obs_obs) / torch.norm(g_obs_obs).clamp_min(1.0e-6)
                        pred_min = g_pred_obs.min().item()
                        pred_max = g_pred_obs.max().item()
                        obs_min = g_obs_obs.min().item()
                        obs_max = g_obs_obs.max().item()
                        g_pred_lin = g_pred_lin_flat
                        g_obs_lin = g_obs_lin_flat
                        pred_mean_log = None
                        pred_std_log = None
                        obs_mean_log = None
                        obs_std_log = None
                        pred_min_log = None
                        pred_max_log = None
                        obs_min_log = None
                        obs_max_log = None
                    check_cfg = data_cfg.get("surrogate_check", {})
                    if check_cfg.get("enabled", False):
                        rel_max = float(check_cfg.get("rel_l2_max", 0.3))
                        warmup_steps = int(check_cfg.get("warmup_steps", 0))
                        if step >= warmup_steps and rel_l2.item() > rel_max and is_main:
                            print(f"Warning: surrogate mismatch rel_l2={rel_l2.item():.4f} > {rel_max:.4f}")
                loss_hist.append(losses["loss"].item())
                data_hist.append(losses["data_loss"].item())
                rel_hist.append(rel_l2.item())
                loss_avg = sum(loss_hist) / float(len(loss_hist))
                data_avg = sum(data_hist) / float(len(data_hist))
                rel_avg = sum(rel_hist) / float(len(rel_hist))
                denom = max(1.0, float(g_obs[0].numel()))
                loss_norm = losses["loss"].item() / denom
                data_norm = losses["data_loss"].item() / denom
                lr_val = opt.param_groups[0]["lr"] if opt.param_groups else 0.0
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = math.sqrt(grad_norm)
                residual_aux = getattr(model_core, "get_last_residual_aux", lambda: {})() or {}
                hopping_mean = None
                hopping_std = None
                if isinstance(physics_meta, dict) and "hopping" in physics_meta:
                    hop_t = physics_meta["hopping"].detach().float()
                    hopping_mean = float(hop_t.mean().item())
                    hopping_std = float(hop_t.std().item()) if hop_t.numel() > 1 else 0.0
                postfix_parts = [
                    f"dom {str(model_cfg.get('data_loss_domain', 'obs_legacy'))[:4]}",
                    f"loss {losses['loss'].item():.4f}",
                    f"data {losses['data_loss'].item():.4f}",
                    f"fft {losses['fft_loss'].item():.4f}",
                    f"rel {rel_l2.item():.4f}",
                    f"loss_n {loss_norm:.6f}",
                    f"data_n {data_norm:.6f}",
                    f"loss_avg {loss_avg:.4f}",
                    f"data_avg {data_avg:.4f}",
                    f"rel_avg {rel_avg:.4f}",
                    f"lr {lr_val:.2e}",
                    f"gn {grad_norm:.3f}",
                    f"it/s {it_s:.2f}",
                    f"s/s {samples_s:.0f}",
                ]
                if "stats_loss" in losses:
                    postfix_parts.append(f"stats {losses['stats_loss'].item():.4f}")
                if "psd_loss" in losses:
                    postfix_parts.append(f"psd {losses['psd_loss'].item():.4f}")
                if "ms_loss" in losses:
                    postfix_parts.append(f"ms {losses['ms_loss'].item():.4f}")
                if "residual_loss" in losses:
                    postfix_parts.append(f"res {losses['residual_loss'].item():.4f}")
                if "sum_rule_loss" in losses:
                    postfix_parts.append(f"sum {losses['sum_rule_loss'].item():.4f}")
                if "residual_active_frac" in residual_aux:
                    postfix_parts.append(f"ract {float(residual_aux['residual_active_frac']):.3f}")
                if hopping_mean is not None:
                    postfix_parts.append(f"t {hopping_mean:.3f}")
                    postfix_parts.append(f"tstd {hopping_std:.3f}")
                summary = f"Step {step}: " + " | ".join(postfix_parts)
                if step % (log_every * 5) == 0:
                    if log_enabled:
                        msg = (
                            f"Step {step}: "
                            f"Pred Lin [{pred_min:.3f}, {pred_max:.3f}] Log [{pred_min_log:.3f}, {pred_max_log:.3f}] | "
                            f"Obs Lin [{obs_min:.3f}, {obs_max:.3f}] Log [{obs_min_log:.3f}, {obs_max_log:.3f}]"
                        )
                        stats_msg = (
                            f"Step {step}: "
                            f"Pred Log mean/std [{pred_mean_log:.3f}, {pred_std_log:.3f}] | "
                            f"Obs Log mean/std [{obs_mean_log:.3f}, {obs_std_log:.3f}]"
                        )
                        if pbar is not None:
                            pbar.write(summary)
                            pbar.write(msg)
                            pbar.write(stats_msg)
                        elif is_main:
                            print(summary)
                            print(msg)
                            print(stats_msg)
                    else:
                        # Linear only logging
                        msg = (
                            f"Step {step}: "
                            f"Pred Obs [{pred_min:.3f}, {pred_max:.3f}] | "
                            f"Obs  Obs [{obs_min:.3f}, {obs_max:.3f}]"
                        )
                        stats_msg = (
                            f"Step {step}: "
                            f"Pred Lin mean/std [{g_pred_lin.mean().item():.3f}, {g_pred_lin.std().item():.3f}] | "
                            f"Obs Lin mean/std [{g_obs_lin.mean().item():.3f}, {g_obs_lin.std().item():.3f}]"
                        )
                        if pbar is not None:
                            pbar.write(summary)
                            pbar.write(msg)
                            pbar.write(stats_msg)
                        elif is_main:
                            print(summary)
                            print(msg)
                            print(stats_msg)

                elif pbar is not None:
                    pbar.write(summary)
                else:
                    print(summary)
                if pbar is not None:
                    pbar.set_postfix_str("")
            
            next_step = step + 1
            if is_main and next_step % data_cfg.get("ckpt_every", 2000) == 0:
                ckpt_dir = config["paths"]["checkpoints"]
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"latent_green_step_{next_step}.pt")
                torch.save(model_core.state_dict(), ckpt_path)

            if pbar is not None:
                pbar.update(1)
            step += 1
            if step >= max_steps:
                break
    if pbar is not None:
        pbar.close()
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = load_config("gd/configs/default.yaml")
    train_latent_green(cfg)

