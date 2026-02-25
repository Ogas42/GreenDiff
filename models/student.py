from typing import Dict, Any
import torch
import torch.nn as nn
from gd.utils.obs_layout import g_obs_to_model_view, obs_channel_count

class ResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return x + h


class StudentModel(nn.Module):
    """
    Student network that predicts V from g_obs.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.model_cfg = config["student"]["model"]
        self.base = self.model_cfg["base_channels"]
        self.num_blocks = self.model_cfg["num_res_blocks"]
        self.dropout = self.model_cfg.get("dropout", 0.0)
        self.K = self.data_cfg["K"]
        self.obs_channels = int(obs_channel_count(self.data_cfg))
        self.in_conv = nn.Conv2d(self.obs_channels, self.base, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(self.base, self.dropout) for _ in range(self.num_blocks)])
        self.out_conv = nn.Conv2d(self.base, 1, kernel_size=3, padding=1)

    def forward(self, g_obs: torch.Tensor) -> torch.Tensor:
        if g_obs.dim() == 5:
            g_obs = g_obs_to_model_view(g_obs, self.data_cfg)
        h = self.in_conv(g_obs)
        h = self.blocks(h)
        V = self.out_conv(h)
        return V
