
import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gd.data.dataset import GFDataset
from gd.models.student import StudentModel
from gd.inference.teacher_sampler import TeacherSampler
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config

# Fix for OMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def verify_student_training():
    print("Starting Student Training Verification...")
    
    # Load config
    config = load_config("gd/configs/default.yaml")
    
    device = torch.device(config["project"]["device"])
    print(f"Using device: {device}")
    
    # Initialize Dataset (Train split)
    print("Initializing Dataset...")
    dataset = GFDataset(config, split="train")
    # Use config batch size to test memory limits
    batch_size = config["student"]["training"]["batch_size"]
    print(f"Using batch size: {batch_size}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Teacher
    print("Initializing TeacherSampler...")
    teacher = TeacherSampler(config)
    
    # Determine Checkpoint Directory
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    # Helper to find latest checkpoint for a pattern
    def find_latest_ckpt(pattern):
        # 1. Try current run
        if os.path.exists(current_ckpt_dir):
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
            if k.startswith("_orig_mod."):
                new_k = k[len("_orig_mod."):]
            elif k.startswith("module."):
                new_k = k[len("module."):]
            else:
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict

    # Load Teacher Weights
    diff_ckpt = find_latest_ckpt("diffusion_step_*.pt")
    if diff_ckpt:
        print(f"Loading Teacher Diffusion from {diff_ckpt}")
        teacher.diffusion.load_state_dict(normalize_state_dict(torch.load(diff_ckpt, map_location=device, weights_only=True)))
    else:
        print("ERROR: No Diffusion checkpoint found! Cannot verify with trained teacher.")
        # return

    vae_ckpt = find_latest_ckpt("vae_step_*.pt")
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        teacher.vae.load_state_dict(normalize_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True)))
    else:
        print("ERROR: No VAE checkpoint found!")
        # return

    lg_ckpt = find_latest_ckpt("latent_green_step_*.pt")
    if lg_ckpt:
        print(f"Loading Latent Green from {lg_ckpt}")
        teacher.latent_green.load_state_dict(normalize_state_dict(torch.load(lg_ckpt, map_location=device, weights_only=True)))
    else:
        print("ERROR: No Latent Green checkpoint found!")
        # return

    teacher.diffusion.to(device).eval()
    teacher.vae.to(device).eval()
    teacher.condition_encoder.to(device).eval()
    teacher.latent_green.to(device).eval()

    # Initialize Student
    print("Initializing StudentModel...")
    student = StudentModel(config).to(device)
    opt = optim.AdamW(student.parameters(), lr=1e-4)

    # Run 1 Training Step
    print("Running 1 training step...")
    for batch in loader:
        g_obs = batch["g_obs"].to(device)
        
        # Teacher Sampling
        print("  Running Teacher Sampling...")
        try:
            with torch.no_grad():
                V_teach = teacher.sample(g_obs)
            print(f"  Teacher Output Shape: {V_teach.shape}")
        except Exception as e:
            print(f"  Teacher Sampling Failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Student Forward
        print("  Running Student Forward...")
        try:
            V_stu = student(g_obs)
            print(f"  Student Output Shape: {V_stu.shape}")
        except Exception as e:
            print(f"  Student Forward Failed: {e}")
            import traceback
            traceback.print_exc()
            return
            
        # Loss Calculation
        loss = torch.mean((V_stu - V_teach) ** 2)
        print(f"  Loss: {loss.item()}")
        
        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("  Backward pass successful.")
        break

    print("Verification Completed Successfully!")

if __name__ == "__main__":
    verify_student_training()

