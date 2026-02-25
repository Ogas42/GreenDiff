
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gd.models.diffusion import LatentDiffusion
from gd.utils.config_utils import load_config

def test_dit():
    # Load config
    config = load_config("gd/configs/default.yaml")
    
    # Mock config for test
    config["data"] = {"resolution": 64, "K": 1}
    config["vae"] = {"latent_channels": 4, "latent_downsample": 4}
    # Ensure hidden_size matches token_dim or projection is handled
    # My DiT implementation: cond_dim is determined by ConditionEncoder output
    # ConditionEncoder output dim is config["diffusion"]["condition_encoder"]["token_dim"]
    # DiT hidden_size is config["diffusion"]["model"]["hidden_size"]
    # CrossAttention expects query_dim = key_dim usually, or handles projection.
    # torch.nn.MultiheadAttention: embed_dim must be same for q,k,v if kdim/vdim not specified.
    # In my DiTBlock:
    # self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, ...)
    # It assumes key/value have dim `hidden_size`.
    # So `token_dim` must equal `hidden_size`.
    
    config["diffusion"]["model"]["hidden_size"] = 256
    config["diffusion"]["model"]["num_heads"] = 8
    config["diffusion"]["condition_encoder"]["token_dim"] = 256
    config["diffusion"]["condition_encoder"]["base_channels"] = 32
    
    # Initialize model
    model = LatentDiffusion(config)
    print("Model initialized.")
    
    # Dummy inputs
    B = 2
    C = 4
    H = 16 # 64/4
    W = 16
    z_t = torch.randn(B, C, H, W)
    t = torch.randint(0, 1000, (B,))
    cond_input = torch.randn(B, 1, 64, 64) # LDOS input
    
    # Forward
    out = model(z_t, t, cond_input)
    print(f"Output shape: {out.shape}")
    assert out.shape == z_t.shape

if __name__ == "__main__":
    test_dit()

