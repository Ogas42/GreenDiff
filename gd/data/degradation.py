from typing import Any, Dict, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationPipeline(nn.Module):
    """
    Differentiable degradation pipeline for simulated STM LDOS observations.

    Supports both legacy LDOS layouts:
      - `(K, H, W)`
      - `(B, K, H, W)`
    and Phase-1 graphene sublattice-resolved canonical layouts:
      - `(K, 2, H, W)`
      - `(B, K, 2, H, W)`
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.apply_prob = config["apply_prob"]
        self.tip_cfg = config["tip_convolution"]
        self.stripe_cfg = config["stripe_noise"]
        self.drift_cfg = config["drift"]
        self.gaussian_cfg = config.get(
            "gaussian_noise", {"enabled": False, "prob": 0.0, "sigma_scale_range": [0.0, 0.0]}
        )
        self.crop_cfg = config.get("crop", {"enabled": False, "prob": 0.0, "scale_range": [1.0, 1.0]})

    def forward(self, g_ideal: torch.Tensor) -> torch.Tensor:
        g, restore = self._flatten_to_4d(g_ideal)
        if self._apply(self.apply_prob):
            if self.tip_cfg["enabled"] and self._apply(self.tip_cfg["prob"]):
                g = self._tip_convolution(g)
            if self.gaussian_cfg["enabled"] and self._apply(self.gaussian_cfg["prob"]):
                g = self._gaussian_noise(g)
            if self.stripe_cfg["enabled"] and self._apply(self.stripe_cfg["prob"]):
                g = self._stripe_noise(g)
            if self.drift_cfg["enabled"] and self._apply(self.drift_cfg["prob"]):
                g = self._low_freq_drift(g)
            if self.crop_cfg["enabled"] and self._apply(self.crop_cfg["prob"]):
                g = self._crop(g)
        return restore(g)

    def _ensure_4d(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Backward-compatible helper retained for existing callers/tests."""
        if x.dim() == 3:
            return x.unsqueeze(0), True
        if x.dim() == 4:
            return x, False
        raise ValueError(f"Expected rank-3/4 LDOS tensor, got shape {tuple(x.shape)}")

    def _flatten_to_4d(self, x: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """
        Flatten non-spatial LDOS axes into channels so all degradations operate on `(B,C,H,W)`.
        """
        if x.dim() == 3:
            return x.unsqueeze(0), lambda y: y.squeeze(0)

        if x.dim() == 4:
            # Ambiguous between legacy batch `(B,C,H,W)` and canonical `(K,2,H,W)`.
            # Canonical layout always has sublattice axis = 2 at dim=1.
            if x.shape[1] == 2:
                k, s, h, w = x.shape
                flat = x.reshape(1, k * s, h, w)
                return flat, lambda y, k=k, s=s, h=h, w=w: y.reshape(k, s, h, w)
            return x, lambda y: y

        if x.dim() == 5:
            b, k, s, h, w = x.shape
            if s != 2:
                raise ValueError(
                    f"Expected canonical sublattice LDOS shape (B,K,2,H,W), got {tuple(x.shape)}"
                )
            flat = x.reshape(b, k * s, h, w)
            return flat, lambda y, b=b, k=k, s=s, h=h, w=w: y.reshape(b, k, s, h, w)

        raise ValueError(f"Unsupported LDOS rank for degradation: shape {tuple(x.shape)}")

    def _tip_convolution(self, x: torch.Tensor) -> torch.Tensor:
        sigma_min, sigma_max = self.tip_cfg["sigma_range"]
        kernel_truncate = self.tip_cfg["kernel_truncate"]
        sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(1, device=x.device).item()
        radius = int(kernel_truncate * sigma)
        kernel_size = 2 * radius + 1
        coords = torch.arange(kernel_size, device=x.device) - radius
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        c = x.shape[1]
        weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=radius, groups=c)

    def _stripe_noise(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.stripe_cfg["mode"]
        scale_min, scale_max = self.stripe_cfg["sigma_scale_range"]
        corr_sigma = self.stripe_cfg["row_corr_sigma"]
        b, c, h, w = x.shape
        per_energy = self.stripe_cfg["per_energy"]
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(self.stripe_cfg["std_floor"])
        amp = scale_min + (scale_max - scale_min) * torch.rand(b, 1, 1, 1, device=x.device)
        if per_energy:
            amp = amp.expand(b, c, 1, 1)
        noise = torch.randn(b, 1 if not per_energy else c, h, 1, device=x.device)
        if corr_sigma > 0:
            radius = int(self.stripe_cfg["kernel_truncate"] * corr_sigma)
            kernel_size = 2 * radius + 1
            coords = torch.arange(kernel_size, device=x.device) - radius
            kernel = torch.exp(-(coords**2) / (2.0 * corr_sigma**2))
            kernel = kernel / kernel.sum()
            weight = kernel.view(1, 1, kernel_size, 1).repeat(noise.shape[1], 1, 1, 1)
            noise = F.conv2d(noise, weight, padding=(radius, 0), groups=noise.shape[1])
        stripes = noise * amp * std
        if mode == "row_bias":
            return x + stripes.expand(-1, c, -1, w)
        return x + stripes.expand(-1, c, -1, w)

    def _low_freq_drift(self, x: torch.Tensor) -> torch.Tensor:
        amp_min, amp_max = self.drift_cfg["alpha_range"]
        cutoff_min, cutoff_max = self.drift_cfg["lowpass_cutoff_range"]
        cutoff = cutoff_min + (cutoff_max - cutoff_min) * torch.rand(1, device=x.device).item()
        multiplicative = self.drift_cfg["multiplicative"]
        b, c, h, w = x.shape
        noise = torch.randn(b, c, h, w, device=x.device)
        fy = torch.fft.fftfreq(h, device=x.device)
        fx = torch.fft.rfftfreq(w, device=x.device)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(kx**2 + ky**2)
        mask = (radius <= cutoff).to(x.dtype)
        spectrum = torch.fft.rfft2(noise, dim=(-2, -1))
        filtered = torch.fft.irfft2(spectrum * mask, s=(h, w), dim=(-2, -1))
        amp = amp_min + (amp_max - amp_min) * torch.rand(b, c, 1, 1, device=x.device)
        if multiplicative:
            return x * (1.0 + filtered * amp)
        return x + filtered * amp

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        scale_min, scale_max = self.gaussian_cfg["sigma_scale_range"]
        std = x.std(dim=(-2, -1), keepdim=True)
        sigma = scale_min + (scale_max - scale_min) * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        return x + torch.randn_like(x) * sigma * std

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        scale_min, scale_max = self.crop_cfg["scale_range"]
        _, _, h, w = x.shape
        scale = scale_min + (scale_max - scale_min) * torch.rand(1, device=x.device).item()
        h_new = max(1, int(h * scale))
        w_new = max(1, int(w * scale))
        top = torch.randint(0, h - h_new + 1, (1,), device=x.device).item() if h > h_new else 0
        left = torch.randint(0, w - w_new + 1, (1,), device=x.device).item() if w > w_new else 0
        crop = x[:, :, top : top + h_new, left : left + w_new]
        return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)

    def _apply(self, prob: float) -> bool:
        return torch.rand(1).item() < prob
