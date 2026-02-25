from typing import Dict, Any, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .latent_green import LatentGreen
from .condition_encoder import ConditionEncoder
from gd.utils.obs_layout import g_obs_to_model_view

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0, use_cross_attn: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.use_cross_attn = use_cross_attn
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        
        # AdaLN modulation: shift, scale, gate for norm1/attn and norm2/mlp
        # We produce 6 params: shift1, scale1, gate1, shift2, scale2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb, cond=None):
        # x: (B, N, D)
        # t_emb: (B, D)
        # cond: (B, L, D) - condition tokens
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        
        # Self Attention
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h, h, h)[0]
        x = x + gate_msa.unsqueeze(1) * h
        
        if self.use_cross_attn and cond is not None:
            h_cross = self.norm_cross(x)
            h_cross = self.cross_attn(h_cross, cond, cond)[0]
            x = x + h_cross
        
        # MLP
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t_emb):
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class LatentDiffusion(nn.Module):
    """
    DiT (Diffusion Transformer) with Cross Attention for Physical Guidance.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_cfg = config["data"]
        self.vae_cfg = config["vae"]
        self.diff_cfg = config["diffusion"]
        self.model_cfg = self.diff_cfg["model"]
        self.train_cfg = self.diff_cfg.get("training", {})
        
        self.T = self.diff_cfg["T"]
        self.schedule = self.diff_cfg["schedule"]
        self.prediction_type = self.train_cfg.get("prediction_type", "eps")
        
        # Dimensions
        self.in_channels = self.vae_cfg["latent_channels"]
        self.patch_size = self.model_cfg["patch_size"]
        self.hidden_size = self.model_cfg["hidden_size"]
        self.num_heads = self.model_cfg["num_heads"]
        self.depth = self.model_cfg["depth"]
        self.mlp_ratio = self.model_cfg.get("mlp_ratio", 4.0)
        self.dropout = self.model_cfg.get("dropout", 0.1)
        self.cond_mode = self.model_cfg.get("cond_mode", "cross_attn")
        self.use_green_attn = self.model_cfg.get("use_green_attention", False) and self.cond_mode != "concat"
        
        # Latent spatial size
        self.H = self.data_cfg["resolution"] // self.vae_cfg["latent_downsample"]
        self.W = self.H # Assume square
        assert self.H % self.patch_size == 0, "Latent size must be divisible by patch size"
        self.num_patches_h = self.H // self.patch_size
        self.num_patches_w = self.W // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Condition Encoder
        self.condition_encoder = ConditionEncoder(config)
        cond_channels = self.condition_encoder.out_channels
        cond_enc_mode = self.diff_cfg["condition_encoder"].get("mode", "token")
        if self.cond_mode == "concat" and cond_enc_mode != "map":
            raise ValueError(
                f"LatentDiffusion cond_mode='concat' requires condition_encoder.mode='map', got {cond_enc_mode!r}"
            )
        if self.cond_mode != "concat" and cond_enc_mode == "map":
            raise ValueError(
                "LatentDiffusion cross-attention modes require token-like condition encoder output "
                "(condition_encoder.mode != 'map')."
            )
        
        # Components
        in_channels = self.in_channels + cond_channels if self.cond_mode == "concat" else self.in_channels
        self.x_embedder = nn.Conv2d(in_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Physics Operator (Green's Function)
        if self.use_green_attn:
            self.latent_green = LatentGreen(config)
            # Freeze latent green as it is a pre-trained physical prior
            for p in self.latent_green.parameters():
                p.requires_grad = False
        
        # Learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.hidden_size), requires_grad=True)
        
        # Blocks
        use_cross_attn = self.cond_mode != "concat"
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, self.dropout, use_cross_attn=use_cross_attn)
            for _ in range(self.depth)
        ])
        
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.in_channels
        p = self.patch_size
        h = self.num_patches_h
        w = self.num_patches_w
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Noisy latent (B, C, H, W)
            t: Time (B,)
            cond_input: Raw condition tensor (B, C_in, H_img, W_img) or canonical sublattice LDOS
                (B, K, 2, H_img, W_img).
        """
        if cond_input.dim() == 5:
            cond_input = g_obs_to_model_view(cond_input, self.data_cfg)
        if self.cond_mode == "concat":
            if cond_input.dim() == 4 and cond_input.shape[1] == self.condition_encoder.out_channels and cond_input.shape[2] == self.H and cond_input.shape[3] == self.W:
                cond_map = cond_input
            else:
                cond_map = self.condition_encoder(cond_input)
            if cond_map.shape[2] != z_t.shape[2] or cond_map.shape[3] != z_t.shape[3]:
                cond_map = F.interpolate(cond_map, size=z_t.shape[2:], mode="bilinear", align_corners=False)
            x = self.x_embedder(torch.cat([z_t, cond_map], dim=1))
            cond = None
        else:
            if cond_input.dim() == 3:
                cond = cond_input
            else:
                cond = self.condition_encoder(cond_input)
            if self.use_green_attn:
                g_pred = self.latent_green(z_t, t)
                cond_pred = self.condition_encoder(g_pred)
                cond = torch.cat([cond, cond_pred], dim=1)
            x = self.x_embedder(z_t)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        x = x + self.pos_embed
        
        # Embed time
        t_emb = self.t_embedder(t) # (B, D)
        
        # Transformer Blocks
        half = self.depth // 2
        skips = []
        for i, block in enumerate(self.blocks):
            if i >= half:
                idx = half - 1 - (i - half)
                if idx >= 0 and idx < len(skips):
                    x = (x + skips[idx]) / 1.414
            x = block(x, t_emb, cond)
            if i < half:
                skips.append(x)
            
        # Final layer
        x = self.final_layer(x, t_emb) # (B, N, p*p*C)
        
        # Unpatchify
        return self.unpatchify(x)

    def predict_eps(self, z_t: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor) -> torch.Tensor:
        pred = self.forward(z_t, t, cond_input)
        if self.prediction_type == "v":
            alpha_t, sigma_t = self.get_alpha_sigma(t)
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            return sigma_t * z_t + alpha_t * pred
        if self.prediction_type == "x0":
            alpha_t, sigma_t = self.get_alpha_sigma(t)
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            sigma_t = sigma_t.view(-1, 1, 1, 1)
            return (z_t - alpha_t * pred) / sigma_t.clamp_min(1.0e-6)
        return pred

    def predict_x0(self, z_t: torch.Tensor, t: torch.Tensor, cond_input: torch.Tensor) -> torch.Tensor:
        pred = self.forward(z_t, t, cond_input)
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        if self.prediction_type == "v":
            return alpha_t * z_t - sigma_t * pred
        if self.prediction_type == "x0":
            return pred
        return (z_t - sigma_t * pred) / alpha_t.clamp_min(1.0e-6)

    def step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_input: torch.Tensor,
        eta: float,
        t_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps = self.predict_eps(z_t, t, cond_input)
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        if t_prev is None:
            t_prev = (t - 1).clamp_min(0)
        alpha_prev, sigma_prev = self.get_alpha_sigma(t_prev)
        
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        alpha_prev = alpha_prev.view(-1, 1, 1, 1)
        sigma_prev = sigma_prev.view(-1, 1, 1, 1)
        
        x0 = (z_t - sigma_t * eps) / alpha_t.clamp_min(1.0e-6)
        if eta > 0:
            # DDIM stochasticity term. eta=0 recovers deterministic DDIM update.
            sigma_eta = eta * torch.sqrt(
                ((sigma_prev ** 2) / (sigma_t ** 2).clamp_min(1.0e-12))
                * (1.0 - (alpha_t ** 2) / (alpha_prev ** 2).clamp_min(1.0e-12))
            ).clamp_min(0.0)
            dir_coeff = torch.sqrt((sigma_prev ** 2 - sigma_eta ** 2).clamp_min(0.0))
            z_prev = alpha_prev * x0 + dir_coeff * eps + sigma_eta * torch.randn_like(z_t)
        else:
            z_prev = alpha_prev * x0 + sigma_prev * eps
        return z_prev

    def get_alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t.clamp(min=0, max=max(0, self.T - 1)).float() / float(self.T)
        if self.schedule == "cosine":
            s = 0.008
            alpha = torch.cos((t + s) / (1 + s) * math.pi / 2)
        else:
            alpha = 1.0 - t
        alpha = alpha.clamp_min(1.0e-5)
        sigma = torch.sqrt(1.0 - alpha**2)
        return alpha, sigma
