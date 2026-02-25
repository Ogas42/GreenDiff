import os
import sys
import glob
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.data.dataset import GFDataset
from gd.models.vae import VAE
from gd.utils.config_utils import get_latest_checkpoint_dir, load_config, resolve_config_paths

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def find_latest_ckpt(runs_root: str, current_ckpt_dir: str, pattern: str):
    if os.path.exists(current_ckpt_dir):
        ckpts = sorted(glob.glob(os.path.join(current_ckpt_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            return ckpts[-1]
    latest_dir = get_latest_checkpoint_dir(runs_root, require_pattern=pattern)
    if latest_dir:
        ckpts = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if ckpts:
            print(f"Found {pattern} in latest run: {latest_dir}")
            return ckpts[-1]
    return None

def normalize_state_dict(state_dict):
    keys = list(state_dict.keys())
    if keys and all(k.startswith("_orig_mod.") for k in keys):
        state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def main():
    parser = argparse.ArgumentParser(description="Test VAE reconstruction")
    parser.add_argument("--config", default="gd/configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory to load weights from")
    parser.add_argument("--allow_mismatch", action="store_true", help="Allow loading incompatible checkpoints")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(config["project"]["device"])
    runs_root = config.get("paths", {}).get("runs_root", config["paths"]["workdir"])
    current_ckpt_dir = config["paths"]["checkpoints"]
    if args.ckpt_dir:
        current_ckpt_dir = args.ckpt_dir
        run_dir = os.path.dirname(current_ckpt_dir)
        runs_root = os.path.dirname(run_dir)
        config["paths"]["checkpoints"] = current_ckpt_dir
        config["paths"]["workdir"] = run_dir
        config["paths"]["runs_root"] = runs_root
        config = resolve_config_paths(config)
    print(f"Searching for checkpoints. Current: {current_ckpt_dir}, Root: {runs_root}")

    vae = VAE(config).to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    vae_ckpt = find_latest_ckpt(runs_root, current_ckpt_dir, "vae_step_*.pt")
    if vae_ckpt:
        print(f"Loading VAE from {vae_ckpt}")
        state_dict = torch.load(vae_ckpt, map_location=device, weights_only=True)
        state_dict = normalize_state_dict(state_dict)
        model_keys = set(vae.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing_keys = sorted(model_keys - ckpt_keys)
        unexpected_keys = sorted(ckpt_keys - model_keys)
        if missing_keys or unexpected_keys:
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            if not args.allow_mismatch:
                print("Checkpoint incompatible with current VAE. Use --allow_mismatch to proceed.")
                return
        vae.load_state_dict(state_dict, strict=False)
    else:
        print("Warning: No VAE checkpoint found!")

    dataset_root = config.get("paths", {}).get("dataset_root", "data_cache")
    has_val_cache = os.path.exists(os.path.join(dataset_root, "val.pt")) or len(glob.glob(os.path.join(dataset_root, "val_shard_*.pt"))) > 0
    if not has_val_cache:
        print(f"No cached validation data found in {dataset_root}. Please generate cache before running this test.")
        return

    dataset = GFDataset(config, split="val")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    V = batch["V"].to(device)
    V_in = V.unsqueeze(1) if V.dim() == 3 else V

    with torch.no_grad():
        z, mu, logvar = vae.encode(V_in)
        z_eval = mu if mu is not None else z
        V_hat = vae.decode(z_eval)
        losses = vae.loss(V_in, V_hat, mu, logvar)

    print(f"V range: [{V_in.min().item():.4f}, {V_in.max().item():.4f}]")
    print(f"V_hat range: [{V_hat.min().item():.4f}, {V_hat.max().item():.4f}]")
    print(f"loss: {losses['loss'].item():.6f}")
    print(f"recon: {losses['recon_loss'].item():.6f}")
    print(f"kl: {losses['kl_loss'].item():.6f}")

    n_samples = min(4, V_in.shape[0])
    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]
    for i in range(n_samples):
        v_true = V_in[i, 0].detach().cpu().numpy()
        v_rec = V_hat[i, 0].detach().cpu().numpy()
        axes[i, 0].imshow(v_true, cmap="inferno")
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(v_rec, cmap="inferno")
        axes[i, 1].set_title("VAE Reconstruction")
        axes[i, 1].axis("off")
    save_path = "vae_test_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()

