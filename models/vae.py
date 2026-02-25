from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class VAE(nn.Module):
    """
    Latent potential model for AE/VAE.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vae_cfg = config["vae"]
        self.data_cfg = config["data"]
        self.mode = self.vae_cfg["mode"]
        self.latent_downsample = self.vae_cfg["latent_downsample"]
        self.latent_channels = self.vae_cfg["latent_channels"]
        self.base_channels = self.vae_cfg["encoder"]["base_channels"]
        self.num_res_blocks = self.vae_cfg["encoder"].get("num_res_blocks", 2)
        self.dropout = self.vae_cfg["encoder"].get("dropout", 0.0)
        self.dec_base_channels = self.vae_cfg.get("decoder", {}).get("base_channels", self.base_channels)
        self.dec_num_res_blocks = self.vae_cfg.get("decoder", {}).get("num_res_blocks", self.num_res_blocks)
        self.dec_dropout = self.vae_cfg.get("decoder", {}).get("dropout", self.dropout)
        self.resolution = self.data_cfg["resolution"]
        self.h = self.resolution // self.latent_downsample
        self.w = self.resolution // self.latent_downsample
        if self.latent_channels not in (4, 8):
            raise ValueError("latent_channels must be 4 or 8")
        if self.resolution not in (64, 256):
            raise ValueError("resolution must be 64 or 256")
        if self.latent_downsample not in (2, 4):
            raise ValueError("latent_downsample must be 2 or 4")

        enc_out_channels = self.latent_channels * (2 if self.mode == "vae" else 1)
        down_steps = int(math.log2(self.latent_downsample))
        channel_mults = [2] if down_steps == 1 else [1, 2]

        enc_blocks = [
            nn.Conv2d(1, self.base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(self.num_res_blocks):
            enc_blocks.append(ResBlock(self.base_channels, self.dropout))
        current_ch = self.base_channels
        for mult in channel_mults:
            out_ch = self.base_channels * mult
            enc_blocks.append(nn.Conv2d(current_ch, out_ch, kernel_size=4, stride=2, padding=1))
            enc_blocks.append(nn.SiLU())
            for _ in range(self.num_res_blocks):
                enc_blocks.append(ResBlock(out_ch, self.dropout))
            current_ch = out_ch
        enc_blocks.append(nn.Conv2d(current_ch, enc_out_channels, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*enc_blocks)

        dec_blocks = [
            nn.Conv2d(self.latent_channels, self.dec_base_channels * channel_mults[-1], kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        for _ in range(self.dec_num_res_blocks):
            dec_blocks.append(ResBlock(self.dec_base_channels * channel_mults[-1], self.dec_dropout))
        current_ch = self.dec_base_channels * channel_mults[-1]
        rev_mults = list(reversed(channel_mults))
        for idx, mult in enumerate(rev_mults):
            next_mult = rev_mults[idx + 1] if idx + 1 < len(rev_mults) else 1
            out_ch = self.dec_base_channels * next_mult
            dec_blocks.append(nn.ConvTranspose2d(current_ch, out_ch, kernel_size=4, stride=2, padding=1))
            dec_blocks.append(nn.SiLU())
            for _ in range(self.dec_num_res_blocks):
                dec_blocks.append(ResBlock(out_ch, self.dec_dropout))
            current_ch = out_ch
        dec_blocks.append(nn.Conv2d(current_ch, 1, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            V: Potential of shape (B, 1, H, W).
        Returns:
            Tuple of (z, mu, logvar), where z is (B, C, h, w).
        """
        if V.dim() == 3:
            V = V.unsqueeze(1)
        h = self.encoder(V)
        if self.mode == "vae":
            mu, logvar = torch.chunk(h, 2, dim=1)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        return h, None, None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (B, C, h, w).
        Returns:
            torch.Tensor: Reconstructed potential of shape (B, 1, H, W).
        """
        return self.decoder(z)

    def forward(self, V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            V: Potential of shape (B, 1, H, W).
        Returns:
            Tuple of (V_hat, mu, logvar).
        """
        z, mu, logvar = self.encode(V)
        return self.decode(z), mu, logvar

    def loss(self, V: torch.Tensor, V_hat: torch.Tensor, mu: Optional[torch.Tensor], logvar: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            V: Target potential of shape (B, 1, H, W).
            V_hat: Reconstruction of shape (B, 1, H, W).
        Returns:
            Dict with keys: loss, recon_loss, kl_loss.
        """
        recon_type = self.vae_cfg.get("recon_loss_type", "l1")
        if recon_type == "mse":
            recon = F.mse_loss(V_hat, V)
        elif recon_type == "log_cosh":
            eps = self.vae_cfg.get("recon_log_cosh_eps", 1.0e-6)
            diff = V_hat - V
            recon = torch.log(torch.cosh(diff) + eps).mean()
        else:
            recon = F.l1_loss(V_hat, V)
        if self.mode == "vae" and mu is not None and logvar is not None:
            kl = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)
            kl_weight = self.vae_cfg["kl"]["weight"]
        else:
            kl = torch.zeros((), device=V.device)
            kl_weight = 0.0
        return {"loss": recon + kl_weight * kl, "recon_loss": recon, "kl_loss": kl}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
