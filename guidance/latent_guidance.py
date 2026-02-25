from typing import Any, Dict

import torch

from gd.utils.ldos_transform import ldos_obs_from_linear
from gd.utils.obs_layout import g_obs_to_model_view


class LatentGuidance:
    """
    Physics-based latent guidance using a frozen LatentGreen surrogate.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.guidance_cfg = config["guidance"]
        self.lambda_cfg = self.guidance_cfg["lambda"]
        self.loss_cfg = self.guidance_cfg["loss"]
        self.data_cfg = config.get("data", {})

    def apply(
        self,
        z: torch.Tensor,
        g_obs: torch.Tensor,
        t: torch.Tensor,
        diffusion_model: torch.nn.Module,
        latent_green: torch.nn.Module,
    ) -> torch.Tensor:
        z = z.detach().requires_grad_(True)
        g_pred = latent_green(z, t)
        loss = self._loss(g_pred, g_obs)
        grad = torch.autograd.grad(loss, z, create_graph=False)[0]
        alpha, sigma = diffusion_model.get_alpha_sigma(t)
        schedule = self.lambda_cfg.get("schedule", "sigma2")
        if schedule == "late_strong":
            lam = self.lambda_cfg["lambda0"] * (1.0 - alpha) ** 2
        else:
            lam = self.lambda_cfg["lambda0"] * (sigma**2)
        lam = lam.view(-1, 1, 1, 1).to(z.device)
        z_updated = z - lam * grad
        return z_updated.detach()

    def _loss(self, g_pred: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        g_pred = ldos_obs_from_linear(g_pred, self.data_cfg)
        if g_obs.dim() == 5:
            g_obs = g_obs_to_model_view(g_obs, self.data_cfg)
        loss_type = self.loss_cfg["type"]
        if loss_type == "obs_consistency":
            return torch.mean((g_pred - g_obs) ** 2)
        if loss_type == "charbonnier":
            eps = self.loss_cfg["charbonnier_eps"]
            return torch.mean(torch.sqrt((g_pred - g_obs) ** 2 + eps**2))
        if loss_type == "huber":
            delta = self.loss_cfg["huber_delta"]
            diff = g_pred - g_obs
            return torch.mean(
                torch.where(torch.abs(diff) < delta, 0.5 * diff**2, delta * (torch.abs(diff) - 0.5 * delta))
            )
        return torch.mean((g_pred - g_obs) ** 2)
