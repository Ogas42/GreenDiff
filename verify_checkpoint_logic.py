
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config

def verify():
    config = load_config("gd/configs/default.yaml")
    work_dir = config["paths"]["runs_root"]
    
    print(f"Checking runs in {work_dir}")
    
    # Test 1: Find latest run with any checkpoints (default behavior)
    latest_any = get_latest_checkpoint_dir(work_dir)
    print(f"Latest run (any): {latest_any}")
    
    # Test 2: Find latest run with VAE checkpoints
    latest_vae = get_latest_checkpoint_dir(work_dir, require_pattern="vae_step_*.pt")
    print(f"Latest run (VAE): {latest_vae}")
    
    # Test 3: Find latest run with Latent Green checkpoints
    latest_lg = get_latest_checkpoint_dir(work_dir, require_pattern="latent_green_step_*.pt")
    print(f"Latest run (Latent Green): {latest_lg}")

    # Test 4: Find latest run with Diffusion checkpoints
    latest_diff = get_latest_checkpoint_dir(work_dir, require_pattern="diffusion_step_*.pt")
    print(f"Latest run (Diffusion): {latest_diff}")
    
    # Test 5: Find latest run with a NON-EXISTENT pattern
    latest_none = get_latest_checkpoint_dir(work_dir, require_pattern="non_existent_*.pt")
    print(f"Latest run (Non-existent): {latest_none}")

if __name__ == "__main__":
    verify()

