import os
import sys
import argparse
import datetime
import shutil
import glob
import yaml
import importlib
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.data.dataset import generate_cache
from gd.utils.config_utils import load_config, resolve_config_paths
from gd.utils.ldos_transform import force_linear_ldos_mode

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    @property
    def encoding(self):
        for s in self.streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"

def get_timestamp_dir(base_dir: str) -> str:
    """Generate a timestamped directory name."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(base_dir, timestamp)

def save_config(config: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def copy_checkpoints(src_dir: str, dst_dir: str, patterns: List[str]):
    """Copy checkpoints matching patterns from src to dst."""
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist. Skipping copy.")
        return

    os.makedirs(dst_dir, exist_ok=True)
    for pattern in patterns:
        files = glob.glob(os.path.join(src_dir, pattern))
        for f in files:
            print(f"Copying {os.path.basename(f)} to {dst_dir}")
            shutil.copy2(f, dst_dir)

def interactive_selection() -> List[str]:
    """Interactive prompt to select stages."""
    stages = []
    print("\n=== GreenDiff Pipeline Selection ===")
    print("Please select stages to run (y/n):")
    
    if input("1. Generate Data Cache? [y/N]: ").lower().startswith('y'):
        stages.append("data")
    
    if input("2. Train VAE? [y/N]: ").lower().startswith('y'):
        stages.append("vae")
    
    if input("3. Train Latent Green? [y/N]: ").lower().startswith('y'):
        stages.append("green")
        
    if input("4. Train Diffusion? [y/N]: ").lower().startswith('y'):
        stages.append("diffusion")
        
    if input("5. Train Student? [y/N]: ").lower().startswith('y'):
        stages.append("student")
        
    return stages

def load_train_fn(module_name: str, fn_name: str):
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None:
        available = [k for k in dir(module) if not k.startswith("_")]
        raise ImportError(f"{fn_name} not found in {module_name}. Available: {available}")
    return fn

def main():
    parser = argparse.ArgumentParser(description="Run GreenDiff Pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--stages", nargs="+", choices=["data", "vae", "green", "diffusion", "student"], 
                        help="Stages to run (space separated)")
    parser.add_argument("--init_from", help="Path to previous run directory to copy checkpoints from")
    parser.add_argument("--no_interactive", action="store_true", help="Disable interactive mode")
    
    args = parser.parse_args()

    # Determine stages
    if args.stages:
        stages = args.stages
    elif not args.no_interactive:
        stages = interactive_selection()
    else:
        print("No stages selected. Exiting.")
        return

    if not stages:
        print("No stages selected. Exiting.")
        return

    print(f"\nSelected stages: {', '.join(stages)}")

    # Load Config
    config = load_config(args.config)
    force_linear_ldos_mode(config, verbose=True, context="run_pipeline")
    
    # Setup Directories
    base_runs_dir = config["paths"]["runs_root"]
    run_dir = get_timestamp_dir(base_runs_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    
    print(f"Creating run directory: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    pipeline_log_path = os.path.join(log_dir, "pipeline.log")

    # Update Config
    # Important: Store the original runs root so training scripts can find previous runs!
    config["paths"]["runs_root"] = base_runs_dir
    config["paths"]["workdir"] = run_dir
    config["paths"]["checkpoints"] = ckpt_dir
    config["paths"]["logs"] = log_dir
    config = resolve_config_paths(config)
    
    # Save run config
    save_config(config, os.path.join(run_dir, "config.yaml"))

    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    log_fp = open(pipeline_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(stdout_backup, log_fp)
    sys.stderr = TeeStream(stderr_backup, log_fp)
    print(f"Saving pipeline log to: {pipeline_log_path}")

    # Copy checkpoints if requested
    try:
        if args.init_from:
            src_ckpt = os.path.join(args.init_from, "checkpoints")
            print(f"\nInitializing from {src_ckpt}...")
            # Copy everything just in case
            copy_checkpoints(src_ckpt, ckpt_dir, ["*.pt"])
        
        # Execute Stages
        try:
            if "data" in stages:
                print("\n" + "="*50)
                print("Running Stage 0: Data Generation")
                print("="*50)
                generate_cache(config, ["train", "val"])

            if "vae" in stages:
                train_vae = load_train_fn("gd.train.train_vae", "train_vae")
                print("\n" + "="*50)
                print("Running Stage 1: VAE Training")
                print("="*50)
                train_vae(config)
                
            if "green" in stages:
                train_latent_green = load_train_fn("gd.train.train_latent_green", "train_latent_green")
                print("\n" + "="*50)
                print("Running Stage 2: Latent Green Training")
                print("="*50)
                train_latent_green(config)
                eval_mod = importlib.import_module("gd.test.test_green")
                argv_backup = sys.argv
                config_path = os.path.join(run_dir, "config.yaml")
                sys.argv = ["test_green.py", "--config", config_path]
                eval_mod.main()
                sys.argv = argv_backup
                
            if "diffusion" in stages:
                train_diffusion = load_train_fn("gd.train.train_diffusion", "train_diffusion")
                print("\n" + "="*50)
                print("Running Stage 3: Diffusion Training")
                print("="*50)
                train_diffusion(config)
                
            if "student" in stages:
                train_student = load_train_fn("gd.train.train_student", "train_student")
                print("\n" + "="*50)
                print("Running Stage 4: Student Training")
                print("="*50)
                train_student(config)
                
            print(f"\nPipeline completed successfully! Results saved to {run_dir}")
            
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup
        log_fp.close()

if __name__ == "__main__":
    main()

