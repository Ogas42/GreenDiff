from typing import Dict, Any
import torch
import torch.nn as nn
from gd.utils.obs_layout import g_obs_to_model_view, is_sublattice_resolved, obs_channel_count

class ConditionEncoder(nn.Module):
    """
    Encodes external measurements/conditions into an embedding space.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.enc_cfg = config["diffusion"]["condition_encoder"]
        self.K = self.data_cfg["K"]
        self.obs_channels = int(obs_channel_count(self.data_cfg))
        self.sublattice_resolved_ldos = bool(is_sublattice_resolved(self.data_cfg))
        self.embed_dim = self.enc_cfg["token_dim"]
        self.mode = self.enc_cfg.get("mode", "token")
        self.use_coords = self.enc_cfg.get("use_coords", False) # Add coordinates to input
        self.latent_channels = self.enc_cfg.get("latent_channels", config["vae"]["latent_channels"])
        base = self.enc_cfg["base_channels"]
        layers = self.enc_cfg["num_layers"]
        blocks = []
        if self.mode == "energy_seq":
            if self.sublattice_resolved_ldos:
                raise ValueError(
                    "diffusion.condition_encoder.mode='energy_seq' is not supported with "
                    "data.sublattice_resolved_ldos=true in Phase 1."
                )
            in_ch = 1
        else:
            in_ch = self.obs_channels
            if self.use_coords:
                in_ch += 2 # Add x, y channels
        for i in range(layers):
            out_ch = base * (2 ** min(i, 2))
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            blocks.append(nn.GroupNorm(8, out_ch))
            blocks.append(nn.SiLU())
            # Only downsample if we haven't reached the target downsample ratio yet
            # Target is VAE downsample (usually 4)
            # Current implementation downsamples at every layer except last
            # We need to be smarter.
            current_ds = 2 ** (i + 1)
            target_ds = config["vae"]["latent_downsample"]
            
            if i < layers - 1:
                stride = 2 if current_ds <= target_ds else 1
                blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3 if stride==1 else 4, stride=stride, padding=1))
                blocks.append(nn.GroupNorm(8, out_ch))
                blocks.append(nn.SiLU())
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.proj = nn.Linear(in_ch, self.embed_dim)
        self.map_proj = nn.Conv2d(in_ch, self.latent_channels, kernel_size=1)
        self.out_channels = self.latent_channels if self.mode == "map" else self.embed_dim
        scale_init = float(self.enc_cfg.get("scale", 1.0))
        self.learnable_scale = bool(self.enc_cfg.get("learnable_scale", False))
        if self.learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        else:
            self.scale = scale_init

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: Raw measurement tensor of shape (B, C_in, H, W).
        Returns:
            torch.Tensor: Condition tokens of shape (B, L, E).
        """
        if condition.dim() == 5:
            condition = g_obs_to_model_view(condition, self.data_cfg)
        if self.enc_cfg.get("normalize", False):
            eps = self.enc_cfg.get("norm_eps", 1.0e-6)
            mean = condition.mean(dim=(-2, -1), keepdim=True)
            std = condition.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
            condition = (condition - mean) / std
        
        if self.use_coords and self.mode != "energy_seq":
            B, _, H, W = condition.shape
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=condition.device), 
                                            torch.linspace(-1, 1, W, device=condition.device), indexing="ij")
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
            condition = torch.cat([condition, grid_x, grid_y], dim=1)

        if self.mode == "energy_seq":
            B, K, H, W = condition.shape
            x = condition.view(B * K, 1, H, W)
            h = self.backbone(x)
            h = h.mean(dim=(2, 3))
            out = self.proj(h)
            out = out.view(B, K, self.embed_dim)
            return out * self.scale
        h = self.backbone(condition)
        if self.mode == "map":
            out = self.map_proj(h)
            return out * self.scale
        B, C, H, W = h.shape
        h = h.flatten(2).transpose(1, 2)
        out = self.proj(h)
        return out * self.scale
