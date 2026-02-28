from typing import Dict, Any, Optional, Tuple
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from gd.utils.ldos_transform import (
    ldos_linear_from_obs,
    ldos_log_enabled,
    ldos_log_eps,
    ldos_obs_from_linear,
)
from gd.utils.loss_align import align_pred, per_energy_affine
from gd.utils.obs_layout import (
    flatten_sub_for_energy_ops,
    g_obs_to_canonical_view,
    g_obs_to_model_view,
    is_sublattice_resolved,
    obs_channel_count,
)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float,
        time_embed_dim: Optional[int] = None,
        cond_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, channels))
        self.cond_proj = None
        if cond_embed_dim is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_embed_dim, channels * 2))

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        if self.cond_proj is not None and cond_emb is not None:
            scale, shift = self.cond_proj(cond_emb).chunk(2, dim=-1)
            h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return x + h


def _group_norm_groups(channels: int) -> int:
    groups = min(8, max(1, int(channels)))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _make_feature_norm(channels: int, norm_type: str) -> nn.Module:
    norm = str(norm_type).lower()
    if norm == "identity":
        return nn.Identity()
    if norm == "groupnorm":
        return nn.GroupNorm(_group_norm_groups(channels), channels)
    raise ValueError(f"Unsupported LatentGreen norm_type={norm_type!r}; expected 'groupnorm' or 'identity'.")


def _apply_feature_modulation(
    h: torch.Tensor,
    t_emb: Optional[torch.Tensor],
    time_proj: Optional[nn.Module],
    cond_emb: Optional[torch.Tensor],
    cond_proj: Optional[nn.Module],
) -> torch.Tensor:
    if time_proj is not None and t_emb is not None:
        h = h + time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
    if cond_proj is not None and cond_emb is not None:
        scale, shift = cond_proj(cond_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
    return h


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = max(1, int(modes_x))
        self.modes_y = max(1, int(modes_y))
        self.weight_real = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels, self.modes_x, self.modes_y)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels, self.modes_x, self.modes_y)
        )
        self.reset_parameters()

    def reset_parameters(self):
        scale = 1.0 / math.sqrt(max(1, self.in_channels + self.out_channels))
        nn.init.uniform_(self.weight_real, -scale, scale)
        nn.init.uniform_(self.weight_imag, -scale, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_ft = torch.fft.rfft2(x.to(torch.float32), dim=(-2, -1), norm="ortho")
        out_ft = x_ft.new_zeros((x.shape[0], self.out_channels, x_ft.shape[-2], x_ft.shape[-1]))
        mx = min(self.modes_x, x_ft.shape[-2], self.weight_real.shape[-2])
        my = min(self.modes_y, x_ft.shape[-1], self.weight_real.shape[-1])
        if mx > 0 and my > 0:
            x_sub = x_ft[:, :, :mx, :my]
            xr = x_sub.real
            xi = x_sub.imag
            wr = self.weight_real[:, :, :mx, :my]
            wi = self.weight_imag[:, :, :mx, :my]
            out_real = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
            out_imag = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
            out_ft[:, :, :mx, :my] = torch.complex(out_real, out_imag)
        out = torch.fft.irfft2(out_ft, s=x.shape[-2:], dim=(-2, -1), norm="ortho")
        return out.to(orig_dtype)


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_x: int,
        modes_y: int,
        time_embed_dim: Optional[int] = None,
        cond_embed_dim: Optional[int] = None,
        pointwise_skip: bool = True,
        norm_type: str = "groupnorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1) if pointwise_skip else None
        self.time_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, channels))
        self.cond_proj = None
        if cond_embed_dim is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_embed_dim, channels * 2))
        self.norm = _make_feature_norm(channels, norm_type)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.spectral(x)
        if self.pointwise is not None:
            h = h + self.pointwise(x)
        h = _apply_feature_modulation(h, t_emb, self.time_proj, cond_emb, self.cond_proj)
        h = self.norm(h)
        h = self.dropout(self.act(h))
        return x + h


class HybridFNOBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes_x: int,
        modes_y: int,
        local_branch_channels: int,
        local_branch_depth: int,
        time_embed_dim: Optional[int] = None,
        cond_embed_dim: Optional[int] = None,
        pointwise_skip: bool = True,
        norm_type: str = "groupnorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1) if pointwise_skip else None
        local_layers = []
        local_hidden = int(local_branch_channels)
        local_depth = max(1, int(local_branch_depth))
        in_channels = channels
        for idx in range(local_depth):
            out_channels = channels if idx == local_depth - 1 else local_hidden
            local_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if idx != local_depth - 1:
                local_layers.append(nn.GELU())
            in_channels = local_hidden
        self.local_branch = nn.Sequential(*local_layers)
        self.time_proj = None
        if time_embed_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, channels))
        self.cond_proj = None
        if cond_embed_dim is not None:
            self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_embed_dim, channels * 2))
        self.norm = _make_feature_norm(channels, norm_type)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.spectral(x)
        if self.pointwise is not None:
            h = h + self.pointwise(x)
        h = h + self.local_branch(x)
        h = _apply_feature_modulation(h, t_emb, self.time_proj, cond_emb, self.cond_proj)
        h = self.norm(h)
        h = self.dropout(self.act(h))
        return x + h


class LatentGreen(nn.Module):
    """
    Latent Green operator mapping z to g_pred.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.vae_cfg = config["vae"]
        self.physics_cfg = config["physics"]
        self.model_cfg = config["latent_green"]["model"]
        self.cond_cfg = config["latent_green"].get("conditioning", {})
        self.noise_cfg = config["latent_green"]["noisy_latent_training"]
        self.target_representation = str(self.data_cfg.get("target_representation", "ldos_ab"))
        self.resolution = self.data_cfg["resolution"]
        self.latent_channels = self.vae_cfg["latent_channels"]
        self.latent_downsample = self.vae_cfg["latent_downsample"]
        self.h = self.resolution // self.latent_downsample
        self.w = self.resolution // self.latent_downsample
        if self.latent_downsample not in (1, 2, 4):
            raise ValueError("LatentGreen supports latent_downsample in {1, 2, 4}.")
        self.num_upsamples = 0 if self.latent_downsample == 1 else (1 if self.latent_downsample == 2 else 2)
        self.K = self.data_cfg["K"]
        self.sublattice_resolved_ldos = bool(is_sublattice_resolved(self.data_cfg))
        self.obs_channels = int(obs_channel_count(self.data_cfg))
        self.base_channels = int(self.model_cfg["base_channels"])
        self.backbone_type = str(self.model_cfg.get("backbone", "cnn")).lower()
        if self.backbone_type not in ("cnn", "fno", "hybrid_fno"):
            raise ValueError(
                f"Unsupported latent_green.model.backbone={self.backbone_type!r}; "
                "expected one of {'cnn', 'fno', 'hybrid_fno'}."
            )
        self.hidden_channels = int(self.model_cfg.get("hidden_channels", self.base_channels))
        self.num_res_blocks = int(self.model_cfg["num_res_blocks"])
        self.dropout = float(self.model_cfg["dropout"])
        feature_default_channels = self.base_channels if self.backbone_type == "cnn" else self.hidden_channels
        self.use_timestep = self.model_cfg.get("use_timestep", True)
        self.time_embed_dim = int(self.model_cfg.get("time_embed_dim", feature_default_channels * 4))
        self.time_embed_freq = int(self.model_cfg.get("time_embed_freq", 256))
        self.fno_layers = max(1, int(self.model_cfg.get("fno_layers", 4)))
        self.fno_modes_x = max(1, int(self.model_cfg.get("fno_modes_x", 12)))
        self.fno_modes_y = max(1, int(self.model_cfg.get("fno_modes_y", 12)))
        self.use_coord_grid = bool(self.model_cfg.get("use_coord_grid", True)) and self.backbone_type in ("fno", "hybrid_fno")
        self.coord_channels = 2 if self.use_coord_grid else 0
        self.spectral_dropout = float(self.model_cfg.get("spectral_dropout", self.dropout))
        self.pointwise_skip = bool(self.model_cfg.get("pointwise_skip", True))
        self.norm_type = str(self.model_cfg.get("norm_type", "groupnorm")).lower()
        self.local_branch_channels = int(self.model_cfg.get("local_branch_channels", self.hidden_channels))
        self.local_branch_depth = max(1, int(self.model_cfg.get("local_branch_depth", 2)))
        self.eta = float(self.physics_cfg["kpm"].get("eta", 0.01))
        ham_cfg = self.physics_cfg["hamiltonian"]
        self.lattice_type = str(ham_cfg.get("type", "square_lattice")).lower()
        if self.lattice_type == "honeycomb":
            self.lattice_type = "graphene"
        t_cfg = ham_cfg.get("t", 1.0)
        self._hopping_is_range = False
        if isinstance(t_cfg, (list, tuple)):
            self._hopping_is_range = len(t_cfg) > 1
            self.hopping = float(sum(t_cfg) / len(t_cfg))
        else:
            self.hopping = float(t_cfg)
        self.mu = float(ham_cfg.get("mu", 0.0))
        self.use_physics_meta_conditioning = bool(self.cond_cfg.get("use_physics_meta", False))
        self.cond_scalar_keys = list(self.cond_cfg.get("scalar_keys", ["hopping"]))
        self.cond_embed_dim = int(self.cond_cfg.get("embed_dim", max(16, feature_default_channels)))
        self.cond_inject_mode = str(self.cond_cfg.get("inject_mode", "film"))
        if self.cond_inject_mode != "film":
            raise ValueError(f"Unsupported latent_green.conditioning.inject_mode={self.cond_inject_mode!r}; expected 'film'.")
        self.physics_cond_mlp = None
        if self.use_physics_meta_conditioning:
            self.physics_cond_mlp = nn.Sequential(
                nn.Linear(len(self.cond_scalar_keys), self.cond_embed_dim),
                nn.SiLU(),
                nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
            )
        self._residual_warned = False
        self._missing_physics_meta_warned = False
        self._residual_supported = self.lattice_type in ("square_lattice", "square", "graphene")
        residual_weight_cfg = float(self._physics_loss_weights().get("residual_weight", 0.0))
        if residual_weight_cfg > 0.0 and not self._residual_supported:
            warnings.warn(
                (
                    "LatentGreen residual_loss is enabled but the current Hamiltonian configuration is not "
                    "supported by the implemented residual operator. Residual terms will be zeroed. "
                    f"(lattice_type={self.lattice_type!r})"
                ),
                RuntimeWarning,
            )
            self._residual_warned = True
        energies_cfg = self.data_cfg["energies"]
        if energies_cfg["mode"] == "linspace":
            energies = torch.linspace(energies_cfg["Emin"], energies_cfg["Emax"], self.K)
        else:
            energies = torch.tensor(energies_cfg["list"], dtype=torch.float32)
        self.register_buffer("energies", energies)

        time_dim = self.time_embed_dim if self.use_timestep else None
        cond_dim = self.cond_embed_dim if self.use_physics_meta_conditioning else None
        self.t_embedder = TimestepEmbedder(self.time_embed_dim, self.time_embed_freq) if self.use_timestep else None
        self.in_proj = None
        self.res1 = None
        self.up1 = None
        self.res2 = None
        self.up2 = None
        self.res3 = None
        self.fno_lift = None
        self.fno_blocks = None
        self.fno_upsamples = None
        if self.backbone_type == "cnn":
            self._build_cnn_backbone(time_dim, cond_dim)
            self.feature_channels = self.base_channels
        else:
            self._build_fno_backbone(time_dim, cond_dim)
            self.feature_channels = self.hidden_channels
        self.psi_out = nn.Conv2d(self.feature_channels, self.obs_channels * 2, kernel_size=3, padding=1)
        self.src_out = nn.Conv2d(self.feature_channels, self.obs_channels, kernel_size=3, padding=1)
        self._last_residual_aux: Dict[str, float] = {"residual_active_frac": 1.0}

    def _build_cnn_backbone(self, time_dim: Optional[int], cond_dim: Optional[int]):
        self.in_proj = nn.Conv2d(self.latent_channels, self.base_channels, kernel_size=3, padding=1)
        self.res1 = nn.ModuleList(
            [ResBlock(self.base_channels, self.dropout, time_dim, cond_dim) for _ in range(self.num_res_blocks)]
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
        )
        self.res2 = nn.ModuleList(
            [ResBlock(self.base_channels, self.dropout, time_dim, cond_dim) for _ in range(self.num_res_blocks)]
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
        )
        self.res3 = nn.ModuleList(
            [ResBlock(self.base_channels, self.dropout, time_dim, cond_dim) for _ in range(self.num_res_blocks)]
        )

    def _build_fno_backbone(self, time_dim: Optional[int], cond_dim: Optional[int]):
        self.fno_lift = nn.Conv2d(self.latent_channels + self.coord_channels, self.hidden_channels, kernel_size=1)
        block_cls = HybridFNOBlock2d if self.backbone_type == "hybrid_fno" else FNOBlock2d
        blocks = []
        for _ in range(self.fno_layers):
            if self.backbone_type == "hybrid_fno":
                block = block_cls(
                    self.hidden_channels,
                    self.fno_modes_x,
                    self.fno_modes_y,
                    self.local_branch_channels,
                    self.local_branch_depth,
                    time_dim,
                    cond_dim,
                    self.pointwise_skip,
                    self.norm_type,
                    self.spectral_dropout,
                )
            else:
                block = block_cls(
                    self.hidden_channels,
                    self.fno_modes_x,
                    self.fno_modes_y,
                    time_dim,
                    cond_dim,
                    self.pointwise_skip,
                    self.norm_type,
                    self.spectral_dropout,
                )
            blocks.append(block)
        self.fno_blocks = nn.ModuleList(blocks)
        self.fno_upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                )
                for _ in range(self.num_upsamples)
            ]
        )

    def _build_coord_grid(self, z: torch.Tensor) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=z.shape[-2], device=z.device, dtype=z.dtype)
        x = torch.linspace(-1.0, 1.0, steps=z.shape[-1], device=z.device, dtype=z.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
        return grid.expand(z.shape[0], -1, -1, -1)

    def _forward_cnn_backbone(
        self,
        z: torch.Tensor,
        t_emb: Optional[torch.Tensor],
        cond_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.in_proj(z)
        for block in self.res1:
            h = block(h, t_emb, cond_emb)
        if self.num_upsamples >= 1:
            h = self.up1(h)
            for block in self.res2:
                h = block(h, t_emb, cond_emb)
        if self.num_upsamples >= 2:
            h = self.up2(h)
            for block in self.res3:
                h = block(h, t_emb, cond_emb)
        return h

    def _forward_fno_backbone(
        self,
        z: torch.Tensor,
        t_emb: Optional[torch.Tensor],
        cond_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = z
        if self.use_coord_grid:
            x = torch.cat([x, self._build_coord_grid(z)], dim=1)
        h = self.fno_lift(x)
        for block in self.fno_blocks:
            h = block(h, t_emb, cond_emb)
        for upsample in self.fno_upsamples:
            h = upsample(h)
        return h

    def forward(
        self,
        z: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        physics_meta: Optional[Dict[str, torch.Tensor]] = None,
        defect_meta: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
        return_psi: bool = False,
    ):
        """
        Args:
            z: Latent tensor of shape (B, C, h, w).
            t: Optional diffusion/noise timestep indices of shape (B,).
            physics_meta: Optional per-sample physics metadata (e.g. hopping).
            defect_meta: Reserved for future forward conditioning (unused in Phase 1.5).
            return_aux: When True, return a dict payload for multi-target training.
            return_psi: Legacy API; when True, also return the surrogate complex field and source term.
        Returns:
            `g_pred` is a non-negative surrogate LDOS in physical linear space with shape
            `(B, obs_channels, H, W)` where `obs_channels=K` (legacy) or `2K` (A/B-resolved).
            If `return_psi=True`, returns `(g_pred, psi_real, psi_imag, src)`.
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        t_emb = None
        if self.use_timestep and t is not None:
            t_emb = self.t_embedder(t)
        cond_emb = self._build_physics_cond_embedding(z, physics_meta)
        if self.backbone_type == "cnn":
            h = self._forward_cnn_backbone(z, t_emb, cond_emb)
        else:
            h = self._forward_fno_backbone(z, t_emb, cond_emb)
        psi = self.psi_out(h)
        psi_real, psi_imag = psi.chunk(2, dim=1)
        g_pred = self._ldos_from_psi(psi_real, psi_imag)
        src = self.src_out(h)
        if return_aux:
            return {
                "ldos_lin": g_pred,
                "psi_real": psi_real,
                "psi_imag": psi_imag,
                "src": src,
            }
        if return_psi:
            return g_pred, psi_real, psi_imag, src
        return g_pred

    def add_noise(self, z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Clean latent of shape (B, C, h, w).
            t: Time indices of shape (B,).
        Returns:
            Tuple of (z_t, alpha, sigma).
        """
        alpha, sigma = self._alpha_sigma(t, self.noise_cfg["T"], self.noise_cfg["schedule"])
        alpha = alpha.view(-1, 1, 1, 1).to(z.device)
        sigma = sigma.view(-1, 1, 1, 1).to(z.device)
        eps = torch.randn_like(z)
        z_t = alpha * z + sigma * eps
        return z_t, alpha, sigma

    def loss(
        self,
        g_pred: torch.Tensor | Dict[str, torch.Tensor],
        g_obs: torch.Tensor,
        residual_loss: Optional[torch.Tensor] = None,
        physics_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g_pred: Predicted g in model view, shape `(B,C,H,W)`.
            g_obs: Observed g in dataset canonical view, shape `(B,K,H,W)` or `(B,K,2,H,W)`.
        Notes:
            `g_pred` is expected in linear LDOS space and is internally mapped into observation space.
            `g_obs` is expected in dataset observation space (`data.ldos_transform` applied).
        Returns:
            Dict with keys including:
            `loss`, `data_loss`, `fft_loss`, `psd_loss`, `stats_loss`, `linear_scale_loss`,
            `ms_loss`, `residual_loss`.
        """
        if isinstance(g_pred, dict):
            pred_pack = g_pred
            g_pred = pred_pack.get("ldos_lin")
            if g_pred is None:
                raise ValueError("LatentGreen.loss pred_pack must contain key 'ldos_lin' in Phase 1.5.")
        if g_pred.dim() != 4:
            raise ValueError(f"LatentGreen.loss expects g_pred rank 4 model-view tensor, got {tuple(g_pred.shape)}")
        if self.sublattice_resolved_ldos:
            if g_obs.dim() != 5:
                raise ValueError(
                    f"LatentGreen.loss expects canonical sublattice g_obs rank 5 (B,K,2,H,W), got {tuple(g_obs.shape)}"
                )
            g_obs_flat = flatten_sub_for_energy_ops(g_obs)
        else:
            if g_obs.dim() != 4:
                raise ValueError(f"LatentGreen.loss expects g_obs rank 4, got {tuple(g_obs.shape)}")
            g_obs_flat = g_obs
        if g_pred.shape[0] != g_obs.shape[0] or g_pred.shape[-2:] != g_obs.shape[-2:]:
            raise ValueError(f"LatentGreen.loss batch/spatial mismatch: g_pred={tuple(g_pred.shape)} vs g_obs={tuple(g_obs.shape)}")
        if g_pred.shape[1] != self.obs_channels:
            raise ValueError(f"LatentGreen.loss expected g_pred channels={self.obs_channels}, got {g_pred.shape[1]}")
        eps = ldos_log_eps(self.data_cfg)
        data_loss_domain = str(self.model_cfg.get("data_loss_domain", "obs_legacy")).lower()

        loss_type = self.model_cfg.get("loss_type", "mse")
        huber_beta = float(self.model_cfg.get("huber_beta", 0.1))
        psd_loss_weight = float(self.model_cfg.get("psd_loss_weight", 0.0))
        psd_eps = float(self.model_cfg.get("psd_eps", 1.0e-8))
        use_per_energy_affine = bool(self.model_cfg.get("per_energy_affine", False))
        align_cfg = self.model_cfg.get("energy_align", {})
        align_enabled = bool(align_cfg.get("enabled", False))
        align_max_shift = int(align_cfg.get("max_shift", 0))
        log_cosh_eps = float(self.model_cfg.get("log_cosh_eps", 1.0e-6))

        def _psd_loss(pred, obs):
            pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
            obs_f = torch.fft.rfft2(obs, dim=(-2, -1), norm="ortho")
            pred_p = pred_f.real ** 2 + pred_f.imag ** 2
            obs_p = obs_f.real ** 2 + obs_f.imag ** 2
            pred_l = torch.log(pred_p + psd_eps)
            obs_l = torch.log(obs_p + psd_eps)
            return F.l1_loss(pred_l, obs_l)

        def _reduce_data_loss(pred_x: torch.Tensor, obs_x: torch.Tensor) -> torch.Tensor:
            if loss_type == "log_cosh":
                return torch.log(torch.cosh(pred_x - obs_x) + log_cosh_eps).mean()
            if loss_type == "l1":
                return F.l1_loss(pred_x, obs_x)
            if loss_type == "huber":
                return F.smooth_l1_loss(pred_x, obs_x, beta=huber_beta)
            return F.mse_loss(pred_x, obs_x)

        if data_loss_domain == "linear_normalized":
            pred_lin_c = g_obs_to_canonical_view(g_pred, self.data_cfg) if self.sublattice_resolved_ldos else g_pred.unsqueeze(2)
            obs_lin_raw = ldos_linear_from_obs(g_obs, self.data_cfg).clamp_min(0.0)
            obs_lin_c = obs_lin_raw if self.sublattice_resolved_ldos else obs_lin_raw.unsqueeze(2)
            obs_scale = torch.sqrt((obs_lin_c ** 2).mean(dim=(1, 2, 3, 4), keepdim=True) + 1.0e-12).clamp_min(1.0e-6)
            pred_norm_c = pred_lin_c / obs_scale
            obs_norm_c = obs_lin_c / obs_scale
            if self.sublattice_resolved_ldos:
                pred = flatten_sub_for_energy_ops(pred_norm_c)
                obs = flatten_sub_for_energy_ops(obs_norm_c)
            else:
                pred = pred_norm_c.squeeze(2)
                obs = obs_norm_c.squeeze(2)
            data_loss = _reduce_data_loss(pred, obs)
        elif data_loss_domain == "obs_legacy":
            g_pred_for_loss = ldos_obs_from_linear(g_pred, self.data_cfg)
            pred_lin_c = g_obs_to_canonical_view(g_pred, self.data_cfg) if self.sublattice_resolved_ldos else g_pred.unsqueeze(2)
            obs_lin_raw = ldos_linear_from_obs(g_obs, self.data_cfg).clamp_min(0.0)
            obs_lin_c = obs_lin_raw if self.sublattice_resolved_ldos else obs_lin_raw.unsqueeze(2)
            if self.sublattice_resolved_ldos:
                pred_canonical = g_obs_to_canonical_view(g_pred_for_loss, self.data_cfg)
                pred = flatten_sub_for_energy_ops(pred_canonical)
                obs = g_obs_flat
            else:
                pred = g_pred_for_loss
                obs = g_obs_flat
            if use_per_energy_affine:
                pred = per_energy_affine(pred, obs)
            pred, per_energy_loss = align_pred(
                pred,
                obs,
                enabled=align_enabled,
                max_shift=align_max_shift,
                loss_type=loss_type,
                huber_beta=huber_beta,
                log_cosh_eps=log_cosh_eps,
            )
            data_loss = per_energy_loss.mean()
        else:
            raise ValueError(f"Unsupported latent_green.model.data_loss_domain={data_loss_domain!r}")
        peak_cfg = self.model_cfg.get("peak_control", {})
        if not isinstance(peak_cfg, dict):
            peak_cfg = {}
        peak_enabled = bool(peak_cfg.get("enabled", False))
        log_aux_loss = torch.zeros((), device=g_pred.device)
        topk_peak_loss = torch.zeros((), device=g_pred.device)
        peak_ratio_penalty = torch.zeros((), device=g_pred.device)
        if peak_enabled:
            pred_lin_pos = pred_lin_c.clamp_min(0)
            obs_lin_pos = obs_lin_c.clamp_min(0)
            bsz = pred_lin_pos.shape[0]
            pred_flat = pred_lin_pos.reshape(bsz, -1)
            obs_flat = obs_lin_pos.reshape(bsz, -1)
            peak_eps = 1.0e-6

            # Dynamic-range-compressed auxiliary loss to reduce peak over-shoot.
            log_aux_beta = float(peak_cfg.get("log_aux_huber_beta", 0.1))
            log_scale_mode = str(peak_cfg.get("log_aux_scale", "p95_obs_per_sample"))
            if log_scale_mode == "p95_obs_per_sample":
                scale = torch.quantile(obs_flat, 0.95, dim=1, keepdim=True)
            else:
                scale = obs_flat.mean(dim=1, keepdim=True)
            scale = scale.clamp_min(peak_eps)
            pred_log = torch.log1p(pred_flat / scale)
            obs_log = torch.log1p(obs_flat / scale)
            log_aux_loss = F.smooth_l1_loss(pred_log, obs_log, beta=log_aux_beta)

            # Peak-aware supervision on GT top-k and predicted top-k locations.
            topk_frac = float(peak_cfg.get("topk_frac", 0.005))
            topk_beta = float(peak_cfg.get("topk_huber_beta", 0.1))
            n = pred_flat.shape[1]
            k = max(1, min(n, int(math.ceil(topk_frac * n))))
            idx_obs = torch.topk(obs_flat, k, dim=1).indices
            idx_pred = torch.topk(pred_flat, k, dim=1).indices
            gather_pred_on_obs = torch.gather(pred_flat, 1, idx_obs)
            gather_obs_on_obs = torch.gather(obs_flat, 1, idx_obs)
            gather_pred_on_pred = torch.gather(pred_flat, 1, idx_pred)
            gather_obs_on_pred = torch.gather(obs_flat, 1, idx_pred)
            topk_gt = F.smooth_l1_loss(gather_pred_on_obs, gather_obs_on_obs, beta=topk_beta)
            topk_pred = F.smooth_l1_loss(gather_pred_on_pred, gather_obs_on_pred, beta=topk_beta)
            topk_peak_loss = topk_gt + 0.5 * topk_pred

            # Hard-ish penalty for pathological max-ratio explosion.
            peak_cap = float(peak_cfg.get("peak_ratio_cap", 4.0))
            peak_ratio = pred_flat.max(dim=1).values / obs_flat.max(dim=1).values.clamp_min(peak_eps)
            peak_ratio_penalty = F.relu(peak_ratio - peak_cap).pow(2).mean()
        if self.model_cfg["use_fft_loss"]:
            fft_loss = self._fft_loss(pred, obs)
            weight = self.model_cfg["fft_loss_weight"]
        else:
            fft_loss = torch.zeros((), device=g_pred.device)
            weight = 0.0
        psd_loss = torch.zeros((), device=g_pred.device)
        if psd_loss_weight > 0:
            psd_loss = _psd_loss(pred, obs)
        stats_weight = self.model_cfg.get("stats_loss_weight", 0.0)
        linear_scale_weight = self.model_cfg.get("linear_scale_loss_weight", 0.0)
        stats_loss = torch.zeros((), device=g_pred.device)
        if stats_weight > 0:
            pred_mean = pred.mean(dim=(2, 3))
            obs_mean = obs.mean(dim=(2, 3))
            pred_std = pred.std(dim=(2, 3))
            obs_std = obs.std(dim=(2, 3))
            stats_loss = F.mse_loss(pred_mean, obs_mean) + F.mse_loss(pred_std, obs_std)
        linear_scale_loss = torch.zeros((), device=g_pred.device)
        if linear_scale_weight > 0:
            pred_lin_for_stats = (
                g_pred.clamp_min(0)
                if not self.sublattice_resolved_ldos
                else flatten_sub_for_energy_ops(g_obs_to_canonical_view(g_pred, self.data_cfg))
            )
            obs_lin_for_stats = ldos_linear_from_obs(g_obs, self.data_cfg).clamp_min(0.0)
            if self.sublattice_resolved_ldos:
                obs_lin_for_stats = flatten_sub_for_energy_ops(obs_lin_for_stats)
            if ldos_log_enabled(self.data_cfg):
                pred_mean_lin = pred_lin_for_stats.mean(dim=(2, 3))
                obs_mean_lin = obs_lin_for_stats.mean(dim=(2, 3))
                linear_scale_loss = F.mse_loss(torch.log(pred_mean_lin + eps), torch.log(obs_mean_lin + eps))
            else:
                pred_mean_lin = pred_lin_for_stats.mean(dim=(2, 3))
                obs_mean_lin = obs_lin_for_stats.mean(dim=(2, 3))
                linear_scale_loss = F.mse_loss(pred_mean_lin, obs_mean_lin)
        ms_weight = self.model_cfg.get("multiscale_loss_weight", 0.0)
        ms_loss = torch.zeros((), device=g_pred.device)
        if ms_weight > 0:
            scales = self.model_cfg.get("multiscale_scales", [2, 4])
            ms_terms = []
            for s in scales:
                if pred.shape[-1] >= s and pred.shape[-2] >= s:
                    pred_s = F.avg_pool2d(pred, kernel_size=s, stride=s)
                    obs_s = F.avg_pool2d(obs, kernel_size=s, stride=s)
                    if loss_type == "log_cosh":
                        eps = self.model_cfg.get("log_cosh_eps", 1.0e-6)
                        ms_terms.append(torch.log(torch.cosh(pred_s - obs_s) + eps).mean())
                    elif loss_type == "l1":
                        ms_terms.append(F.l1_loss(pred_s, obs_s))
                    elif loss_type == "huber":
                        ms_terms.append(F.smooth_l1_loss(pred_s, obs_s, beta=huber_beta))
                    else:
                        ms_terms.append(F.mse_loss(pred_s, obs_s))
            if ms_terms:
                ms_loss = torch.stack(ms_terms).mean()
            else:
                ms_loss = data_loss
        if residual_loss is None:
            residual_loss = torch.zeros((), device=g_pred.device)
        phy_w = self._physics_loss_weights()
        sum_rule_loss = self._sum_rule_loss(g_pred, target=phy_w["sum_rule_target"])
        nonneg_loss = torch.zeros((), device=g_pred.device, dtype=g_pred.dtype)
        total = (
            phy_w["data_weight"] * data_loss
            + weight * fft_loss
            + psd_loss_weight * psd_loss
            + stats_weight * stats_loss
            + linear_scale_weight * linear_scale_loss
            + ms_weight * ms_loss
            + phy_w["residual_weight"] * residual_loss
            + phy_w["sum_rule_weight"] * sum_rule_loss
            + phy_w["nonneg_weight"] * nonneg_loss
        )
        return {
            "loss": total,
            "data_loss": data_loss,
            "fft_loss": fft_loss,
            "psd_loss": psd_loss,
            "stats_loss": stats_loss,
            "linear_scale_loss": linear_scale_loss,
            "ms_loss": ms_loss,
            "residual_loss": residual_loss,
            "sum_rule_loss": sum_rule_loss,
            "nonneg_loss": nonneg_loss,
            "log_aux_loss": log_aux_loss,
            "topk_peak_loss": topk_peak_loss,
            "peak_ratio_penalty": peak_ratio_penalty,
            "data_loss_domain": data_loss_domain,
        }

    def _alpha_sigma(self, t: torch.Tensor, T: int, schedule: str) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.float() / float(T)
        if schedule == "cosine":
            s = 0.008
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            # Use sqrt(alpha_bar_t) as the signal coefficient for z_t = alpha * z + sigma * eps.
            alpha = torch.sqrt(f.clamp_min(1.0e-12))
        else:
            alpha = 1.0 - t
        alpha = alpha.clamp_min(1.0e-5)
        sigma = torch.sqrt(1.0 - alpha**2)
        return alpha, sigma

    def _fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        tgt_f = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        pred_amp = torch.abs(pred_f) + 1.0e-8
        tgt_amp = torch.abs(tgt_f) + 1.0e-8
        return F.l1_loss(torch.log(pred_amp), torch.log(tgt_amp))

    def _physics_loss_weights(self) -> Dict[str, float]:
        phy_cfg = self.model_cfg.get("physics_losses", {})
        if not isinstance(phy_cfg, dict):
            phy_cfg = {}
        return {
            "data_weight": float(phy_cfg.get("data_weight", 1.0)),
            "residual_weight": float(phy_cfg.get("residual_weight", self.model_cfg.get("residual_loss_weight", 0.0))),
            "sum_rule_weight": float(phy_cfg.get("sum_rule_weight", 0.0)),
            "nonneg_weight": float(phy_cfg.get("nonneg_weight", 0.0)),
            "kk_weight": float(phy_cfg.get("kk_weight", 0.0)),
            "symmetry_weight": float(phy_cfg.get("symmetry_weight", 0.0)),
            "sum_rule_target": float(phy_cfg.get("sum_rule_target", 1.0)),
        }

    def _build_physics_cond_embedding(
        self,
        z: torch.Tensor,
        physics_meta: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if not self.use_physics_meta_conditioning:
            return None
        vals = []
        for key in self.cond_scalar_keys:
            vals.append(self._extract_scalar_from_physics_meta(key, z, physics_meta))
        x = torch.stack(vals, dim=-1)
        return self.physics_cond_mlp(x)

    def _extract_scalar_from_physics_meta(
        self,
        key: str,
        z: torch.Tensor,
        physics_meta: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if physics_meta is not None and key in physics_meta and physics_meta[key] is not None:
            v = physics_meta[key]
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, device=z.device)
            v = v.to(device=z.device, dtype=z.dtype)
            return v.view(v.shape[0]) if v.dim() > 0 else v.expand(z.shape[0])
        if key == "hopping":
            if self.training and not self._missing_physics_meta_warned:
                warnings.warn(
                    "LatentGreen conditioning is enabled but batch is missing physics_meta['hopping']; "
                    "falling back to config/default hopping. Pass physics_meta for physically consistent random-t training.",
                    RuntimeWarning,
                )
                self._missing_physics_meta_warned = True
            return torch.full((z.shape[0],), float(self.hopping), device=z.device, dtype=z.dtype)
        raise KeyError(f"Unsupported/missing physics_meta scalar key: {key!r}")

    def _extract_hopping_batch(
        self,
        ref: torch.Tensor,
        physics_meta: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if physics_meta is not None and isinstance(physics_meta, dict) and "hopping" in physics_meta:
            hop = physics_meta["hopping"]
            if not isinstance(hop, torch.Tensor):
                hop = torch.as_tensor(hop)
            hop = hop.to(device=ref.device, dtype=ref.dtype)
            if hop.dim() == 0:
                hop = hop.expand(ref.shape[0])
            return hop.view(-1, 1, 1, 1)
        return torch.full((ref.shape[0], 1, 1, 1), float(self.hopping), device=ref.device, dtype=ref.dtype)

    def _get_defect_tensor(
        self,
        defect_meta: Optional[Dict[str, torch.Tensor]],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if defect_meta is None or not isinstance(defect_meta, dict) or key not in defect_meta:
            return None
        x = defect_meta.get(key)
        if x is None:
            return None
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.to(device=device, dtype=dtype)

    def _shift_src(self, x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        out = torch.zeros_like(x)
        H, W = x.shape[-2], x.shape[-1]
        src_x0 = max(0, dx)
        src_x1 = min(H, H + dx)
        dst_x0 = max(0, -dx)
        dst_x1 = min(H, H - dx)
        src_y0 = max(0, dy)
        src_y1 = min(W, W + dy)
        dst_y0 = max(0, -dy)
        dst_y1 = min(W, W - dy)
        if src_x1 <= src_x0 or src_y1 <= src_y0 or dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return out
        out[..., dst_x0:dst_x1, dst_y0:dst_y1] = x[..., src_x0:src_x1, src_y0:src_y1]
        return out

    def get_last_residual_aux(self) -> Dict[str, float]:
        return dict(self._last_residual_aux)

    def _ldos_from_psi(self, psi_real: torch.Tensor, psi_imag: torch.Tensor) -> torch.Tensor:
        return psi_real ** 2 + psi_imag ** 2

    def _sum_rule_loss(self, g_pred_lin: torch.Tensor, target: float = 1.0) -> torch.Tensor:
        if g_pred_lin.dim() != 4:
            raise ValueError(f"_sum_rule_loss expects model-view rank4 tensor, got {tuple(g_pred_lin.shape)}")
        if self.sublattice_resolved_ldos:
            rho_c = g_obs_to_canonical_view(g_pred_lin, self.data_cfg)  # (B,K,2,H,W)
        else:
            rho_c = g_pred_lin.unsqueeze(2)  # (B,K,1,H,W)
        e = self.energies.to(device=g_pred_lin.device, dtype=g_pred_lin.dtype)
        rho_int = torch.trapz(rho_c.clamp_min(0), e, dim=1)
        tgt = torch.full_like(rho_int, float(target))
        return F.mse_loss(rho_int, tgt)

    def _apply_hamiltonian(self, psi: torch.Tensor, V: torch.Tensor, hopping_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if V.dim() == 3:
            V = V.unsqueeze(1)
        if V.dim() == 4 and V.shape[1] != 1:
            V = V[:, :1]
        hop = hopping_batch if hopping_batch is not None else torch.full(
            (psi.shape[0], 1, 1, 1), float(self.hopping), device=psi.device, dtype=psi.dtype
        )
        pad = F.pad(psi, (1, 1, 1, 1), mode="constant", value=0.0)
        up = pad[:, :, :-2, 1:-1]
        down = pad[:, :, 2:, 1:-1]
        left = pad[:, :, 1:-1, :-2]
        right = pad[:, :, 1:-1, 2:]
        neighbor_sum = up + down + left + right
        onsite = (4.0 * hop + V - self.mu)
        return onsite * psi - hop * neighbor_sum

    def _apply_graphene_hamiltonian(
        self,
        psi_c: torch.Tensor,
        V: torch.Tensor,
        hopping_batch: torch.Tensor,
        defect_meta: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if psi_c.dim() != 5 or psi_c.shape[2] != 2:
            raise ValueError(f"_apply_graphene_hamiltonian expects canonical psi (B,K,2,H,W), got {tuple(psi_c.shape)}")
        if V.dim() == 3:
            V = V.unsqueeze(1)
        if V.dim() == 4 and V.shape[1] != 1:
            V = V[:, :1]
        V = V.to(device=psi_c.device, dtype=psi_c.dtype)
        B, K, _, H, W = psi_c.shape
        psi_a = psi_c[:, :, 0]
        psi_b = psi_c[:, :, 1]

        vac = self._get_defect_tensor(defect_meta, "vacancy_mask", device=psi_c.device, dtype=psi_c.dtype)
        if vac is None:
            active_a = torch.ones((B, H, W), device=psi_c.device, dtype=psi_c.dtype)
            active_b = torch.ones((B, H, W), device=psi_c.device, dtype=psi_c.dtype)
        else:
            active_a = (vac[:, 0] < 0.5).to(dtype=psi_c.dtype)
            active_b = (vac[:, 1] < 0.5).to(dtype=psi_c.dtype)
        psi_a_eff = psi_a * active_a.unsqueeze(1)
        psi_b_eff = psi_b * active_b.unsqueeze(1)

        onsite_ab = self._get_defect_tensor(defect_meta, "onsite_ab_delta", device=psi_c.device, dtype=psi_c.dtype)
        if onsite_ab is None:
            onsite_ab = torch.zeros((B, 2, H, W), device=psi_c.device, dtype=psi_c.dtype)
        bond_mod = self._get_defect_tensor(defect_meta, "bond_mod", device=psi_c.device, dtype=psi_c.dtype)
        if bond_mod is None:
            bond_mod = torch.zeros((B, 3, H, W), device=psi_c.device, dtype=psi_c.dtype)
        bond_alive = (bond_mod > -1.0 + 1.0e-8).to(dtype=psi_c.dtype)
        bond_scale = (1.0 + bond_mod).clamp_min(0.0) * bond_alive
        t_bond = hopping_batch * bond_scale  # (B,3,H,W)

        # A(i,j) neighbors: B(i,j), B(i-1,j), B(i,j-1) anchored at A(i,j)
        b0 = psi_b_eff
        b1 = self._shift_src(psi_b_eff, -1, 0)
        b2 = self._shift_src(psi_b_eff, 0, -1)
        a_neighbor = (
            t_bond[:, 0].unsqueeze(1) * b0
            + t_bond[:, 1].unsqueeze(1) * b1
            + t_bond[:, 2].unsqueeze(1) * b2
        )

        # B(i,j) neighbors use bonds anchored on the corresponding A cells:
        # A(i,j) [k=0], A(i+1,j) [k=1], A(i,j+1) [k=2].
        a0 = psi_a_eff
        a1 = self._shift_src(psi_a_eff, +1, 0)
        a2 = self._shift_src(psi_a_eff, 0, +1)
        tb0 = t_bond[:, 0].unsqueeze(1)
        tb1 = self._shift_src(t_bond[:, 1].unsqueeze(1), +1, 0)
        tb2 = self._shift_src(t_bond[:, 2].unsqueeze(1), 0, +1)
        b_neighbor = tb0 * a0 + tb1 * a1 + tb2 * a2

        onsite_a = (V[:, :1] + onsite_ab[:, 0:1] - self.mu)
        onsite_b = (V[:, :1] + onsite_ab[:, 1:2] - self.mu)
        hpsi_a = onsite_a * psi_a_eff - a_neighbor
        hpsi_b = onsite_b * psi_b_eff - b_neighbor
        hpsi_a = hpsi_a * active_a.unsqueeze(1)
        hpsi_b = hpsi_b * active_b.unsqueeze(1)
        return torch.stack([hpsi_a, hpsi_b], dim=2)

    def residual_loss(
        self,
        psi_real: torch.Tensor,
        psi_imag: torch.Tensor,
        src: torch.Tensor,
        V: torch.Tensor,
        physics_meta: Optional[Dict[str, torch.Tensor]] = None,
        defect_meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not self._residual_supported:
            if not self._residual_warned:
                warnings.warn(
                    (
                        "LatentGreen.residual_loss is only implemented for square/graphene Hamiltonians. "
                        "Returning zero residual to avoid applying an inconsistent physics constraint "
                        f"(lattice_type={self.lattice_type!r})."
                    ),
                    RuntimeWarning,
                )
                self._residual_warned = True
            return torch.zeros((), device=psi_real.device, dtype=psi_real.dtype)
        hopping_batch = self._extract_hopping_batch(psi_real, physics_meta)
        eta_batch = None
        if physics_meta is not None and "eta" in physics_meta:
            eta_t = physics_meta["eta"]
            if not isinstance(eta_t, torch.Tensor):
                eta_t = torch.as_tensor(eta_t)
            eta_batch = eta_t.to(device=psi_real.device, dtype=psi_real.dtype).view(-1, 1, 1, 1)
        else:
            eta_batch = torch.full((psi_real.shape[0], 1, 1, 1), float(self.eta), device=psi_real.device, dtype=psi_real.dtype)

        if self.lattice_type in ("graphene",):
            if not self.sublattice_resolved_ldos:
                raise ValueError("Graphene residual requires data.sublattice_resolved_ldos=true in Phase 1.5.")
            psi_r_c = g_obs_to_canonical_view(psi_real, self.data_cfg)
            psi_i_c = g_obs_to_canonical_view(psi_imag, self.data_cfg)
            src_c = g_obs_to_canonical_view(src, self.data_cfg)
            E = self.energies.view(1, -1, 1, 1, 1).to(device=psi_real.device, dtype=psi_real.dtype)
            Hpsi_r_c = self._apply_graphene_hamiltonian(psi_r_c, V, hopping_batch, defect_meta)
            Hpsi_i_c = self._apply_graphene_hamiltonian(psi_i_c, V, hopping_batch, defect_meta)
            eta5 = eta_batch.unsqueeze(1)
            r_real_c = E * psi_r_c - eta5 * psi_i_c - Hpsi_r_c - src_c
            r_imag_c = E * psi_i_c + eta5 * psi_r_c - Hpsi_i_c
            vac = self._get_defect_tensor(defect_meta, "vacancy_mask", device=psi_real.device, dtype=psi_real.dtype)
            if vac is not None:
                active = (vac < 0.5).to(dtype=psi_real.dtype).unsqueeze(1)  # (B,1,2,H,W)
            else:
                active = torch.ones((psi_real.shape[0], 1, 2, psi_real.shape[-2], psi_real.shape[-1]), device=psi_real.device, dtype=psi_real.dtype)
            r2 = (r_real_c ** 2 + r_imag_c ** 2) * active
            denom = active.sum().clamp_min(1.0) * float(self.K)
            self._last_residual_aux = {
                "residual_active_frac": float(active.mean().detach().item()),
            }
            return r2.sum() / denom

        E = self.energies.view(1, -1, 1, 1).to(psi_real.device)
        Hpsi_real = self._apply_hamiltonian(psi_real, V, hopping_batch=hopping_batch)
        Hpsi_imag = self._apply_hamiltonian(psi_imag, V, hopping_batch=hopping_batch)
        r_real = E * psi_real - eta_batch * psi_imag - Hpsi_real - src
        r_imag = E * psi_imag + eta_batch * psi_real - Hpsi_imag
        self._last_residual_aux = {"residual_active_frac": 1.0}
        return (r_real ** 2 + r_imag ** 2).mean()

