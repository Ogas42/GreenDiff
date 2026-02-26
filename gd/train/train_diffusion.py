import torch
import math
import os
import glob
import copy
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any
from gd.data.dataset import GFDataset
from gd.models.vae import VAE
from gd.models.diffusion import LatentDiffusion
from gd.models.latent_green import LatentGreen
from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_obs_from_linear
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import (
    aggregate_sublattice_ldos,
    flatten_sub_for_energy_ops,
    g_obs_to_canonical_view,
    is_sublattice_resolved,
)
from gd.trainers.diffusion_validation import render_diffusion_comparison_grid

# Fix for OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def train_diffusion(config: Dict[str, Any]):
    """
    Main training loop for the Latent Diffusion model.
    """
    print("Starting Latent Diffusion training...")
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
    force_linear_ldos_mode(config, verbose=is_main, context="train_diffusion")
    train_cfg = config["diffusion"]["training"]
    show_progress_bar = bool(train_cfg.get("show_progress_bar", True))
    energy_weights_cfg = train_cfg.get("energy_weights", [])
    if isinstance(energy_weights_cfg, (list, tuple)):
        energy_weights = list(energy_weights_cfg)
    else:
        energy_weights = []
    energy_weight_mode = train_cfg.get("energy_weight_mode", "uniform")
    energy_weight_eps = float(train_cfg.get("energy_weight_eps", 1.0e-6))
    energy_weight_power = float(train_cfg.get("energy_weight_power", 1.0))
    topk_cfg = train_cfg.get("topk_phys", {})
    topk_enabled = bool(topk_cfg.get("enabled", False))
    topk_k = int(topk_cfg.get("k", 0) or 0)
    phys_loss_type = train_cfg.get("phys_loss_type", "mse")
    huber_beta = float(train_cfg.get("huber_beta", 0.1))
    psd_loss_weight = float(train_cfg.get("psd_loss_weight", 0.0))
    psd_eps = float(train_cfg.get("psd_eps", 1.0e-8))
    use_per_energy_affine = bool(train_cfg.get("per_energy_affine", False))
    align_cfg = train_cfg.get("energy_align", {})
    align_enabled = bool(align_cfg.get("enabled", False))
    align_max_shift = int(align_cfg.get("max_shift", 0))
    consistency_loss_weight = train_cfg.get("consistency_loss_weight", 0.0)
    if consistency_loss_weight is None:
        consistency_loss_weight = 0.0
    consistency_loss_weight = float(consistency_loss_weight)
    precision = config["project"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_scaler = use_amp and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    
    data_cfg = config.get("data", {})
    sublattice_resolved = bool(is_sublattice_resolved(config))
    dataset = GFDataset(config, split="train")
    shuffle = data_cfg.get("shuffle")
    if shuffle is None:
        shuffle = not dataset.use_shards
    num_workers = data_cfg.get("num_workers", 0)
    if dataset.use_shards:
        shard_workers = data_cfg.get("shard_workers")
        if shard_workers is None:
            num_workers = min(num_workers, 4)
        else:
            num_workers = shard_workers
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if is_distributed else None
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        pin_memory=data_cfg["pin_memory"],
        sampler=sampler,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = data_cfg["persistent_workers"]
        loader_kwargs["prefetch_factor"] = data_cfg.get("prefetch_factor", 2)
    loader = DataLoader(**loader_kwargs)
    
    # Initialize VAE (frozen)
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
        
    # Initialize Diffusion
    model = LatentDiffusion(config).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    lg_ckpt = find_latest_ckpt("latent_green_step_*.pt")
    latent_green = LatentGreen(config).to(device)
    if lg_ckpt:
        if is_main:
            print(f"Loading Latent Green from {lg_ckpt}")
        lg_state = normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True))
        try:
            latent_green.load_state_dict(lg_state)
            if config["diffusion"]["model"].get("use_green_attention", False):
                model_core.latent_green.load_state_dict(lg_state)
        except RuntimeError as e:
            raise RuntimeError(
                "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the Green stage with schema-v2 cache."
            ) from e
    else:
        if is_main:
            print("Warning: No Latent Green checkpoint found!")
    latent_green.eval()
    for p in latent_green.parameters():
        p.requires_grad = False

    # Resume Diffusion if checkpoint exists
    diff_ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, "diffusion_step_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(diff_ckpts) > 0:
        if is_main:
            print(f"Resuming Diffusion from {diff_ckpts[-1]}")
        try:
            model_core.load_state_dict(normalize_state_dict(torch.load(diff_ckpts[-1], map_location=device, weights_only=True)))
            step = int(diff_ckpts[-1].split("_")[-1].split(".")[0])
            if is_main:
                print(f"Resuming from step {step}")
        except RuntimeError as e:
            if is_main:
                print(f"Warning: Could not load Diffusion checkpoint due to architecture mismatch: {e}")
                print("Starting Diffusion training from scratch with the new SOTA configuration...")
            step = 0
    else:
        if is_main:
            print("Starting Diffusion training from scratch...")
        step = 0
    vae_core = vae
    lg_core = latent_green

    base_lr = train_cfg["lr"]
    opt = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=train_cfg["weight_decay"])
    ema_cfg = train_cfg.get("ema", {})
    use_ema = bool(ema_cfg.get("enabled", False))
    ema_model = None
    ema_decay = None
    if use_ema and is_main:
        ema_model = copy.deepcopy(model_core).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_decay = float(ema_cfg.get("decay", 0.9999))
        if len(diff_ckpts) > 0:
            ema_path = diff_ckpts[-1].replace(".pt", "_ema.pt")
            if os.path.exists(ema_path):
                ema_model.load_state_dict(normalize_state_dict(torch.load(ema_path, map_location=device, weights_only=True)))
    min_snr_cfg = train_cfg.get("min_snr", {})
    use_min_snr = bool(min_snr_cfg.get("enabled", False))
    min_snr_gamma = float(min_snr_cfg.get("gamma", 5.0))
    prediction_type = train_cfg.get("prediction_type", "eps")
    x0_loss_weight = float(train_cfg.get("x0_loss_weight", 0.0))
    phys_loss_weight = float(train_cfg.get("phys_loss_weight", 0.0))
    phys_warmup_cfg = train_cfg.get("phys_warmup", {})
    phys_warmup_enabled = bool(phys_warmup_cfg.get("enabled", False))
    phys_warmup_steps = int(phys_warmup_cfg.get("warmup_steps", 0))
    phys_start_ratio = float(phys_warmup_cfg.get("start_ratio", 1.0))
    phys_end_ratio = float(phys_warmup_cfg.get("end_ratio", 1.0))
    latent_scale_cfg = train_cfg.get("latent_scale", {})
    latent_scale_mode = latent_scale_cfg.get("mode", "none")
    latent_target_std = float(latent_scale_cfg.get("target_std", 1.0))
    latent_scale = float(latent_scale_cfg.get("scale", 1.0))
    latent_eps = float(latent_scale_cfg.get("eps", 1.0e-6))
    lr_schedule_cfg = train_cfg.get("lr_schedule", {})
    use_lr_schedule = bool(lr_schedule_cfg.get("enabled", False))
    lr_schedule_type = lr_schedule_cfg.get("type", "cosine")
    warmup_steps = int(lr_schedule_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(lr_schedule_cfg.get("min_lr_ratio", 0.1))
    warmup_min_ratio = float(lr_schedule_cfg.get("warmup_min_ratio", min_lr_ratio))
    
    # Validation Sampler
    # Ensure condition_encoder is on the correct device
    if is_main:
        val_model = ema_model if ema_model is not None else model_core
        cond_enc = val_model.condition_encoder
        val_sampler = TeacherSampler(config, diffusion_model=val_model, vae=vae, condition_encoder=cond_enc, latent_green=latent_green)
        fixed_vis_n = int(train_cfg.get("vis_fixed_n", 4) or 4)
        fixed_vis_batch = None
        if fixed_vis_n > 0:
            fixed_batch = next(iter(loader))
            def _slice_clone_tree(x, n: int):
                if torch.is_tensor(x):
                    return x[:n].clone()
                if isinstance(x, dict):
                    return {k: _slice_clone_tree(v, n) for k, v in x.items()}
                return x
            fixed_vis_batch = {
                "g_obs": fixed_batch["g_obs"][:fixed_vis_n].clone(),
                "V": fixed_batch["V"][:fixed_vis_n].clone(),
            }
            if isinstance(fixed_batch, dict) and "defect_meta" in fixed_batch:
                fixed_vis_batch["defect_meta"] = _slice_clone_tree(fixed_batch["defect_meta"], fixed_vis_n)
    
    max_steps = train_cfg["max_steps"]
    log_every = train_cfg["log_every"]
    grad_clip = train_cfg["grad_clip"]
    smooth_window = int(train_cfg.get("log_smooth_window", 50))
    loss_hist = deque(maxlen=max(1, smooth_window))
    x0_hist = deque(maxlen=max(1, smooth_window))
    phys_hist = deque(maxlen=max(1, smooth_window))
    phys_coeff_hist = deque(maxlen=max(1, smooth_window))
    consistency_hist = deque(maxlen=max(1, smooth_window))

    log_cosh_eps = float(train_cfg.get("log_cosh_eps", 1.0e-6))

    def _psd_loss_per_sample(pred, obs):
        if pred.dim() == 5:
            pred = flatten_sub_for_energy_ops(pred)
        if obs.dim() == 5:
            obs = flatten_sub_for_energy_ops(obs)
        pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        obs_f = torch.fft.rfft2(obs, dim=(-2, -1), norm="ortho")
        pred_p = pred_f.real ** 2 + pred_f.imag ** 2
        obs_p = obs_f.real ** 2 + obs_f.imag ** 2
        pred_l = torch.log(pred_p + psd_eps)
        obs_l = torch.log(obs_p + psd_eps)
        return F.l1_loss(pred_l, obs_l, reduction="none").mean(dim=(1, 2, 3))

    def _normalize_energy_weights(w):
        w = w.clamp_min(energy_weight_eps)
        if energy_weight_power != 1.0:
            w = w ** energy_weight_power
        scale = w.numel() / w.sum().clamp_min(energy_weight_eps)
        return w * scale

    def _energy_weights_from_obs(obs):
        if obs.dim() == 5:
            obs = flatten_sub_for_energy_ops(obs)
        if energy_weight_mode == "snr":
            mean = obs.abs().mean(dim=(0, 2, 3))
            std = obs.std(dim=(0, 2, 3)).clamp_min(energy_weight_eps)
            w = mean / std
            return _normalize_energy_weights(w)
        if energy_weight_mode == "sensitivity":
            w = obs.std(dim=(0, 2, 3)).clamp_min(energy_weight_eps)
            return _normalize_energy_weights(w)
        return None
    
    pbar = tqdm(total=max_steps, initial=step, desc="Training Diffusion") if (is_main and show_progress_bar) else None
    
    while step < max_steps:
        if sampler is not None:
            epoch = step // max(1, len(loader))
            sampler.set_epoch(epoch)
        for batch in loader:
            V = batch["V"].to(device, non_blocking=True)
            g_obs = batch["g_obs"].to(device, non_blocking=True)
            
            with torch.no_grad():
                z, mu, _ = vae_core.encode(V)
                if mu is not None:
                    z = mu
                if latent_scale_mode == "auto":
                    z_std = z.std(dim=(1, 2, 3), keepdim=True).clamp_min(latent_eps)
                    z = z * (latent_target_std / z_std)
                elif latent_scale_mode == "fixed":
                    z = z * latent_scale
                
            # Diffusion training step
            t = torch.randint(0, model_core.T, (z.shape[0],), device=device)
            noise = torch.randn_like(z)
            alpha_t, sigma_t = model_core.get_alpha_sigma(t)
            alpha_t_1d = alpha_t
            sigma_t_1d = sigma_t
            
            # Reshape for broadcasting
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            
            z_t = alpha_t * z + sigma_t * noise
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(z_t, t, g_obs)
            pred = pred.to(torch.float32)
            if prediction_type == "v":
                target = alpha_t * noise - sigma_t * z
            elif prediction_type == "x0":
                target = z
            else:
                target = noise
            
            snr = (alpha_t_1d ** 2) / (sigma_t_1d ** 2).clamp_min(1.0e-8)
            if use_min_snr:
                # anneal gamma for smoother early training
                gamma_eff = min_snr_gamma
                if step < warmup_steps:
                    gamma_eff = min_snr_gamma + (5.0 - min_snr_gamma) * (1.0 - float(step + 1) / float(max(1, warmup_steps)))
                weight_denom = snr + 1.0 if prediction_type == "v" else snr
                weight = torch.minimum(snr, torch.full_like(snr, gamma_eff)) / weight_denom.clamp_min(1.0e-8)
                per_sample = (pred - target) ** 2
                per_sample = per_sample.mean(dim=(1, 2, 3))
                loss = (weight * per_sample).mean()
            else:
                loss = F.mse_loss(pred, target)
            if prediction_type == "v":
                x0_pred = alpha_t * z_t - sigma_t * pred
            elif prediction_type == "x0":
                x0_pred = pred
            else:
                x0_pred = (z_t - sigma_t * pred) / alpha_t.clamp_min(1.0e-6)

            # Always compute x0 monitor metrics so diffusion quality can be tracked
            # even when x0_loss_weight == 0.
            x0_loss_per_sample = F.mse_loss(x0_pred, z, reduction='none').mean(dim=(1, 2, 3))
            x0_sample_weight = alpha_t.view(-1) ** 2
            x0_loss = (x0_loss_per_sample * x0_sample_weight).mean()
            with torch.no_grad():
                raw_x0_loss_val = F.mse_loss(x0_pred, z).item()

            if x0_loss_weight > 0:
                # Weight x0 loss by alpha_t^2 (or SNR) to suppress contributions from high noise steps
                # where reconstruction is unstable.
                loss = loss + x0_loss_weight * x0_loss

            raw_phys_loss_val = None
            raw_const_loss_val = None
            t_zeros = torch.zeros((z.shape[0],), dtype=torch.long, device=z.device)

            def _compute_phys_terms(x0_in):
                g_pred_phys = lg_core(x0_in, t_zeros)
                g_pred_phys_for_loss = ldos_obs_from_linear(g_pred_phys, data_cfg)
                pred_phys = g_obs_to_canonical_view(g_pred_phys_for_loss, data_cfg) if sublattice_resolved else g_pred_phys_for_loss
                obs_phys = g_obs
                if use_per_energy_affine:
                    pred_phys = per_energy_affine(pred_phys, obs_phys)
                pred_phys, per_energy_loss = align_pred(
                    pred_phys,
                    obs_phys,
                    enabled=align_enabled,
                    max_shift=align_max_shift,
                    loss_type=phys_loss_type,
                    huber_beta=huber_beta,
                    log_cosh_eps=log_cosh_eps,
                )
                if len(energy_weights) == g_obs.shape[1]:
                    w = torch.tensor(energy_weights, device=device, dtype=per_energy_loss.dtype)
                    w = _normalize_energy_weights(w).view(1, -1)
                    per_energy_loss = per_energy_loss * w
                else:
                    w_dyn = _energy_weights_from_obs(obs_phys)
                    if w_dyn is not None:
                        per_energy_loss = per_energy_loss * w_dyn.view(1, -1)
                if topk_enabled and topk_k > 0:
                    k_val = min(topk_k, per_energy_loss.shape[1])
                    phys_loss_per_sample = torch.topk(per_energy_loss, k_val, dim=1).values.mean(dim=1)
                else:
                    phys_loss_per_sample = per_energy_loss.mean(dim=1)
                psd_per_sample = torch.zeros_like(phys_loss_per_sample)
                if psd_loss_weight > 0:
                    psd_per_sample = _psd_loss_per_sample(pred_phys, obs_phys)
                return phys_loss_per_sample, psd_per_sample, pred_phys

            phys_coeff = 0.0
            phys_loss_val = None
            should_monitor_phys = is_main and (step % log_every == 0)
            pred_for_cons = None
            phys_sample_weight = alpha_t.view(-1) ** 2

            if phys_loss_weight > 0:
                phys_loss_per_sample, psd_per_sample, pred_for_cons = _compute_phys_terms(x0_pred)
                with torch.no_grad():
                    raw_phys_loss_val = phys_loss_per_sample.mean().item()

                phys_loss = (phys_loss_per_sample * phys_sample_weight).mean()
                total_phys = phys_loss
                if psd_loss_weight > 0:
                    psd_loss = (psd_per_sample * phys_sample_weight).mean()
                    total_phys = total_phys + psd_loss_weight * psd_loss

                phys_coeff = phys_loss_weight
                if phys_warmup_enabled and phys_warmup_steps > 0:
                    ramp = min(float(step + 1) / float(phys_warmup_steps), 1.0)
                    phys_coeff = phys_loss_weight * (phys_start_ratio + (phys_end_ratio - phys_start_ratio) * ramp)
                loss = loss + phys_coeff * total_phys
                phys_loss_val = phys_loss.item()

                if consistency_loss_weight > 0:
                    pred_diff = pred_for_cons[:, 1:] - pred_for_cons[:, :-1]
                    obs_diff = g_obs[:, 1:] - g_obs[:, :-1]
                    const_map = F.mse_loss(pred_diff, obs_diff, reduction='none')
                    reduce_dims = tuple(range(1, const_map.dim()))
                    const_loss_per_sample = const_map.mean(dim=reduce_dims)
                    weighted_const_loss = (const_loss_per_sample * phys_sample_weight).mean()
                    loss = loss + consistency_loss_weight * weighted_const_loss
                    with torch.no_grad():
                        raw_const_loss_val = const_loss_per_sample.mean().item()
            elif should_monitor_phys:
                with torch.no_grad():
                    phys_loss_per_sample, _psd_monitor, pred_for_cons = _compute_phys_terms(x0_pred.detach())
                    raw_phys_loss_val = phys_loss_per_sample.mean().item()
                    pred_diff = pred_for_cons[:, 1:] - pred_for_cons[:, :-1]
                    obs_diff = g_obs[:, 1:] - g_obs[:, :-1]
                    const_map = F.mse_loss(pred_diff, obs_diff, reduction='none')
                    reduce_dims = tuple(range(1, const_map.dim()))
                    const_loss_per_sample = const_map.mean(dim=reduce_dims)
                    raw_const_loss_val = const_loss_per_sample.mean().item()
            loss_hist.append(loss.item())
            x0_hist.append(raw_x0_loss_val)
            phys_hist.append(raw_phys_loss_val)
            consistency_hist.append(raw_const_loss_val)
            phys_coeff_hist.append(phys_coeff)
            
            current_lr = base_lr
            if use_lr_schedule:
                if step < warmup_steps:
                    lr_scale = float(step + 1) / float(max(1, warmup_steps))
                    current_lr = base_lr * max(warmup_min_ratio, lr_scale)
                else:
                    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                    if lr_schedule_type == "cosine":
                        current_lr = base_lr * (min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)))
                    else:
                        current_lr = base_lr
                for group in opt.param_groups:
                    group["lr"] = current_lr

            opt.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            if ema_model is not None and is_main:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model_core.parameters()):
                        ema_p.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
                    for ema_b, b in zip(ema_model.buffers(), model_core.buffers()):
                        ema_b.copy_(b)
            
            if is_main and step % log_every == 0:
                z_mean = z.mean().item()
                z_std = z.std().item()
                snr_mean = snr.mean().item()
                snr_min = snr.min().item()
                snr_max = snr.max().item()
                if use_min_snr:
                    weight_mean = weight.mean().item()
                else:
                    weight_mean = 1.0
                x0_loss_val = x0_loss.item()
                x0_raw_str = f"{raw_x0_loss_val:.6f}" if raw_x0_loss_val is not None else "na"
                x0_w_str = f"{x0_loss_val:.6f}" if x0_loss_val is not None else "na"
                phys_str = f"{raw_phys_loss_val:.6f}" if raw_phys_loss_val is not None else "na"
                loss_avg = sum(loss_hist) / max(1, len(loss_hist))
                x0_vals = [v for v in x0_hist if v is not None]
                phys_vals = [v for v in phys_hist if v is not None]
                x0_avg = sum(x0_vals) / max(1, len(x0_vals)) if x0_vals else None
                phys_avg = sum(phys_vals) / max(1, len(phys_vals)) if phys_vals else None
                x0_avg_str = f"{x0_avg:.6f}" if x0_avg is not None else "na"
                phys_avg_str = f"{phys_avg:.6f}" if phys_avg is not None else "na"
                const_vals = [v for v in consistency_hist if v is not None]
                const_avg = sum(const_vals) / max(1, len(const_vals)) if const_vals else None
                const_avg_str = f"{const_avg:.6f}" if const_avg is not None else "na"
                phys_coeff_val = phys_coeff
                phys_coeff_str = f"{phys_coeff_val:.4f}"
                phys_coeff_vals = [v for v in phys_coeff_hist if v is not None]
                phys_coeff_avg = sum(phys_coeff_vals) / max(1, len(phys_coeff_vals)) if phys_coeff_vals else None
                phys_coeff_avg_str = f"{phys_coeff_avg:.4f}" if phys_coeff_avg is not None else "na"
                const_step_str = f"{raw_const_loss_val:.6f}" if raw_const_loss_val is not None else "na"
                postfix = {
                    "loss": f"{loss.item():.6f}",
                    "lr": f"{current_lr:.2e}",
                    "snr": f"{snr_mean:.2f}",
                    "const": const_avg_str,
                }
                pbar.set_postfix(postfix)
                print(f"[stats] step={step} bs={z.shape[0]} loss={loss.item():.6f} lr={current_lr:.2e} x0_raw={x0_raw_str} x0_w={x0_w_str} phys={phys_str} const={const_step_str} phys_w={phys_coeff_str}")
                print(f"[avg] loss={loss_avg:.6f} x0_raw={x0_avg_str} phys={phys_avg_str} const={const_avg_str} phys_w={phys_coeff_avg_str}")
                print(f"[stats] z={z_mean:.4f}/{z_std:.4f} snr={snr_mean:.2f}/{snr_min:.2f}/{snr_max:.2f} w={weight_mean:.3f}")

            # Save Checkpoints
            next_step = step + 1
            if is_main and next_step % train_cfg.get("ckpt_every", 2000) == 0:
                os.makedirs(current_ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(current_ckpt_dir, f"diffusion_step_{next_step}.pt")
                torch.save(model_core.state_dict(), ckpt_path)
                if ema_model is not None:
                    ema_path = os.path.join(current_ckpt_dir, f"diffusion_step_{next_step}_ema.pt")
                    torch.save(ema_model.state_dict(), ema_path)
                
            # Validation & Visualization
            if is_main and next_step % train_cfg.get("val_every", 2000) == 0:
                print(f"\nStep {next_step}: Running validation...")
                model_core.eval()
                try:
                    if fixed_vis_batch is not None:
                        g_val = fixed_vis_batch["g_obs"].to(device, non_blocking=True)
                        V_true = fixed_vis_batch["V"].to(device, non_blocking=True)
                        defect_meta_val = fixed_vis_batch.get("defect_meta")
                    else:
                        g_val = g_obs
                        V_true = V
                        defect_meta_val = batch.get("defect_meta") if isinstance(batch, dict) else None
                    n_val = min(int(train_cfg.get("vis_fixed_n", 4) or 4), g_val.shape[0])
                    g_val = g_val[:n_val]
                    V_true = V_true[:n_val]
                    
                    with torch.no_grad():
                        if latent_scale_mode == "auto":
                            z_gt, mu_gt, _ = vae_core.encode(V_true)
                            if mu_gt is not None:
                                z_gt = mu_gt
                            z_std_gt = z_gt.std(dim=(1, 2, 3), keepdim=True).clamp_min(latent_eps)
                            val_sampler.unscale_factor = z_std_gt / max(1e-6, latent_target_std)
                        elif latent_scale_mode == "fixed":
                            val_sampler.unscale_factor = 1.0 / latent_scale
                        else:
                            val_sampler.unscale_factor = 1.0
                        V_pred = val_sampler.sample(g_val)
                        
                    save_path = os.path.join(config["paths"]["workdir"], "images", f"val_step_{next_step}.png")
                    defect_meta_vis = None
                    if isinstance(defect_meta_val, dict):
                        defect_meta_vis = {}
                        for k, v in defect_meta_val.items():
                            if torch.is_tensor(v):
                                defect_meta_vis[k] = v[:n_val]
                    render_diffusion_comparison_grid(
                        V_true=V_true[:n_val],
                        V_pred=V_pred[:n_val],
                        g_obs=g_val[:n_val],
                        defect_meta=defect_meta_vis,
                        save_path=save_path,
                        title_prefix="Validation",
                        show_shared_compare=True,
                    )
                    print(f"Validation images saved to {save_path}")
                except Exception as e:
                    print(f"Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    model_core.train()
                
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
    train_diffusion(cfg)

