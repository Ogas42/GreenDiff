import os
import glob
from typing import Dict, Any
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from gd.data.dataset import GFDataset
from gd.models.student import StudentModel
from gd.models.vae import VAE
from gd.models.latent_green import LatentGreen
from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_obs_from_linear
from gd.utils.obs_layout import g_obs_to_model_view, is_sublattice_resolved
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Fix for OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_student(config: Dict[str, Any]):
    """
    Main distillation loop for the Student model.
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
    force_linear_ldos_mode(config, verbose=is_main, context="train_student")
    train_cfg = config["student"]["training"]
    loss_cfg = config["student"]["loss"]
    sublattice_resolved = bool(is_sublattice_resolved(config))

    data_cfg = config.get("data", {})
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

    teacher = TeacherSampler(config)
    
    # Determine Checkpoint Directory
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    if is_main:
        print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    # Load Teacher Weights
    
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

    # 1. Diffusion
    diff_ckpt = find_latest_ckpt("diffusion_step_*.pt")
    if diff_ckpt:
        if is_main:
            print(f"Loading Teacher Diffusion from {diff_ckpt}")
        try:
            teacher.diffusion.load_state_dict(normalize_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True)))
        except RuntimeError as e:
            raise RuntimeError(
                "Diffusion checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the diffusion stage with schema-v2 cache."
            ) from e
    else:
        if is_main:
            print("Warning: No Diffusion checkpoint found! Student will learn from random noise.")

    # 2. VAE (for Teacher and Physics Loss)
    vae_ckpt = find_latest_ckpt("vae_step_*.pt")
    vae_state = None
    if vae_ckpt:
        if is_main:
            print(f"Loading VAE from {vae_ckpt}")
        vae_state = normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
        teacher.vae.load_state_dict(vae_state)
    else:
        if is_main:
            print("Warning: No VAE checkpoint found!")

    # 3. Latent Green (for Teacher and Physics Loss)
    lg_ckpt = find_latest_ckpt("latent_green_step_*.pt")
    lg_state = None
    if lg_ckpt:
        if is_main:
            print(f"Loading Latent Green from {lg_ckpt}")
        lg_state = normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True))
        try:
            teacher.latent_green.load_state_dict(lg_state)
        except RuntimeError as e:
            raise RuntimeError(
                "Latent Green checkpoint is incompatible with Phase-1 sublattice-resolved LDOS channels (K -> 2K). "
                "Re-train the Green stage with schema-v2 cache."
            ) from e
    
    teacher.diffusion.to(device).eval()
    teacher.vae.to(device).eval()
    teacher.condition_encoder.to(device).eval()
    teacher.latent_green.to(device).eval()
    for p in teacher.diffusion.parameters():
        p.requires_grad = False
    for p in teacher.vae.parameters():
        p.requires_grad = False
    for p in teacher.condition_encoder.parameters():
        p.requires_grad = False
    for p in teacher.latent_green.parameters():
        p.requires_grad = False
    student = StudentModel(config).to(device)
    
    # Initialize VAE/LatentGreen for Physics Loss (reuse loaded weights)
    vae = VAE(config).to(device)
    if vae_state is not None:
        vae.load_state_dict(vae_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
        
    latent_green = LatentGreen(config).to(device)
    if lg_state is not None:
        latent_green.load_state_dict(lg_state)
    latent_green.eval()
    for p in latent_green.parameters():
        p.requires_grad = False

    # Resume Student if checkpoint exists
    stu_ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, "student_step_*.pt")), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(stu_ckpts) > 0:
        if is_main:
            print(f"Resuming Student from {stu_ckpts[-1]}")
        student.load_state_dict(normalize_state_dict(torch.load(stu_ckpts[-1], map_location=device, weights_only=True)))
        step = int(stu_ckpts[-1].split("_")[-1].split(".")[0])
        if is_main:
            print(f"Resuming from step {step}")
    else:
        if is_main:
            print("Starting Student training from scratch...")
        step = 0
    student = torch.nn.parallel.DistributedDataParallel(student.to(device), device_ids=[local_rank], output_device=local_rank) if is_distributed else student.to(device)
    latent_green = latent_green.to(device)
    student_core = student.module if isinstance(student, torch.nn.parallel.DistributedDataParallel) else student
    lg_core = latent_green

    opt = optim.AdamW(student.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    max_steps = train_cfg["max_steps"]
    log_every = train_cfg["log_every"]
    grad_clip = train_cfg["grad_clip"]

    pbar = tqdm(total=max_steps, initial=step, desc="Training Student") if is_main else None
    while step < max_steps:
        if sampler is not None:
            epoch = step // max(1, len(loader))
            sampler.set_epoch(epoch)
        for batch in loader:
            g_obs = batch["g_obs"].to(device, non_blocking=True)
            with torch.no_grad():
                V_teach = teacher.sample(g_obs)
            V_stu = student(g_obs)
            imitation_type = loss_cfg["imitation"]["type"]
            if imitation_type == "l2":
                imitation = torch.mean((V_stu - V_teach) ** 2)
            else:
                imitation = torch.mean(torch.abs(V_stu - V_teach))
            loss = loss_cfg["imitation"]["weight"] * imitation

            if loss_cfg["physics"]["enabled"]:
                z_stu, _, _ = vae.encode(V_stu)
                g_pred = lg_core(z_stu)
                g_pred = ldos_obs_from_linear(g_pred, config)
                g_obs_phys = g_obs_to_model_view(g_obs, config) if sublattice_resolved else g_obs
                robust = loss_cfg["physics"]["robust"]
                if robust == "charbonnier":
                    eps = loss_cfg["physics"]["charbonnier_eps"]
                    phys = torch.mean(torch.sqrt((g_pred - g_obs_phys) ** 2 + eps**2))
                elif robust == "huber":
                    delta = loss_cfg["physics"]["huber_delta"]
                    diff = g_pred - g_obs_phys
                    phys = torch.mean(torch.where(torch.abs(diff) < delta, 0.5 * diff**2, delta * (torch.abs(diff) - 0.5 * delta)))
                else:
                    phys = torch.mean((g_pred - g_obs_phys) ** 2)
                warmup = loss_cfg["physics"]["warmup_steps"]
                beta = loss_cfg["physics"]["weight_max"] * min(1.0, step / float(max(1, warmup)))
                loss = loss + beta * phys

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            opt.step()

            if is_main and step % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "imitation": f"{imitation.item():.6f}"
                })
            
            next_step = step + 1
            if is_main and next_step % train_cfg.get("ckpt_every", 2000) == 0:
                ckpt_dir = config["paths"]["checkpoints"]
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"student_step_{next_step}.pt")
                torch.save(student_core.state_dict(), ckpt_path)

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
    train_student(cfg)

