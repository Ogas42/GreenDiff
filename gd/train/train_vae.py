import os
import glob
import sys
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any
from gd.data.dataset import GFVOnlyDataset, ensure_v_only_cache
from gd.models.vae import VAE
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config
from gd.utils.ldos_transform import force_linear_ldos_mode

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def _cfg_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'n', 'off'}:
            return False
    return bool(value)

# Fix for OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_vae(config: Dict[str, Any]):
    """
    Main training loop for the VAE model.
    """
    print("Starting VAE training...")
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
    force_linear_ldos_mode(config, verbose=is_main, context="train_vae")
    train_cfg = config["vae"]["training"]
    precision = config["project"].get("precision", "fp32")
    use_amp = device.type == "cuda" and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_scaler = use_amp and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["project"].get("cudnn_benchmark", False))
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    
    data_cfg = config.get("data", {})
    if is_distributed:
        if is_main:
            ensure_v_only_cache(config, split="train", verbose=True)
        dist.barrier()
    else:
        ensure_v_only_cache(config, split="train", verbose=is_main)
    dataset = GFVOnlyDataset(config, split="train")
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
    
    model = VAE(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config["project"].get("compile", False) and hasattr(torch, "compile") and not is_distributed:
        print("Enabling torch.compile (initial steps may be slow due to compilation)...")
        model = torch.compile(model)

    def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        keys = list(state_dict.keys())
        if keys and all(k.startswith("_orig_mod.") for k in keys):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        keys = list(state_dict.keys())
        if keys and all(k.startswith("module.") for k in keys):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    # Determine Checkpoint Directory & Resume
    work_dir = config["paths"]["workdir"]
    runs_root = config.get("paths", {}).get("runs_root", work_dir)
    current_ckpt_dir = config["paths"]["checkpoints"]

    latest_ckpt_dir = get_latest_checkpoint_dir(runs_root, require_pattern="vae_step_*.pt")
    if latest_ckpt_dir and is_main:
        print(f"Found latest run with VAE checkpoints: {latest_ckpt_dir}")
    elif is_main:
        print("No previous VAE checkpoints found.")

    has_current_ckpt = len(glob.glob(os.path.join(current_ckpt_dir, "vae_step_*.pt"))) > 0
    ckpt_dir = current_ckpt_dir if has_current_ckpt else (latest_ckpt_dir or current_ckpt_dir)
    if is_main:
        print(f"Using checkpoint directory: {ckpt_dir}")

    # Resume VAE if checkpoint exists
    vae_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "vae_step_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(vae_ckpts) > 0:
        if is_main:
            print(f"Resuming VAE from {vae_ckpts[-1]}")
        try:
            state_dict = torch.load(vae_ckpts[-1], map_location=device, weights_only=True)
            state_dict = normalize_state_dict(state_dict)
            load_target = model._orig_mod if hasattr(model, "_orig_mod") else model
            load_target.load_state_dict(state_dict)
            step = int(vae_ckpts[-1].split("_")[-1].split(".")[0])
            if is_main:
                print(f"Resuming from step {step}")
        except RuntimeError as e:
            if is_main:
                print(f"Warning: Could not load checkpoint due to architecture mismatch: {e}")
                print("Starting VAE training from scratch with the new optimized architecture...")
            ckpt_dir = current_ckpt_dir
            if is_main:
                print(f"Switching checkpoint directory to current run: {ckpt_dir}")
            step = 0
    else:
        if is_main:
            print("Starting VAE training from scratch...")
        ckpt_dir = current_ckpt_dir
        if is_main:
            print(f"Using checkpoint directory: {ckpt_dir}")
        step = 0
        
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model_core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    opt = optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    
    max_steps = train_cfg["max_steps"]
    log_every = train_cfg["log_every"]
    grad_clip = train_cfg["grad_clip"]
    
    show_progress_bar = _cfg_bool(train_cfg.get("show_progress_bar", True), default=True)
    if not sys.stderr.isatty():
        show_progress_bar = False
    pbar = tqdm(total=max_steps, initial=step, desc="Training VAE") if (is_main and show_progress_bar) else None
    last_saved_step = 0
    
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

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                V_hat, mu, logvar = model(V)
                losses = model_core.loss(V, V_hat, mu, logvar)
                loss = losses["loss"]

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
            
            if is_main and step % log_every == 0:
                postfix = {
                    "loss": f"{losses['loss'].item():.6f}",
                    "recon": f"{losses['recon_loss'].item():.6f}",
                    "kl": f"{losses['kl_loss'].item():.6f}"
                }
                if pbar is not None:
                    pbar.set_postfix(postfix)
                else:
                    pct = 100.0 * float(step) / float(max(1, max_steps))
                    print(
                        f"[vae] step={step}/{max_steps} ({pct:.1f}%) "
                        f"loss={losses['loss'].item():.6f} recon={losses['recon_loss'].item():.6f} "
                        f"kl={losses['kl_loss'].item():.6f}"
                    )
            
            next_step = step + 1
            if is_main and next_step % train_cfg.get("ckpt_every", 2000) == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"vae_step_{next_step}.pt")
                save_target = model_core._orig_mod if hasattr(model_core, "_orig_mod") else model_core
                torch.save(save_target.state_dict(), ckpt_path)
                last_saved_step = next_step

            if pbar is not None:
                pbar.update(1)
            step += 1
            if step >= max_steps:
                break
    if pbar is not None:
        pbar.close()
    if is_main and step > 0:
        final_step = int(step)
        final_ckpt_path = os.path.join(ckpt_dir, f"vae_step_{final_step}.pt")
        if final_step != last_saved_step and not os.path.exists(final_ckpt_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            save_target = model_core._orig_mod if hasattr(model_core, "_orig_mod") else model_core
            torch.save(save_target.state_dict(), final_ckpt_path)
            print(f"Saved final VAE checkpoint to {final_ckpt_path}")
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = load_config("gd/configs/default.yaml")
    train_vae(cfg)

