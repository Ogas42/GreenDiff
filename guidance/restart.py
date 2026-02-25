from typing import Dict, Any, Tuple
import torch
from gd.data.kpm_forward import KPMForward
from gd.models.vae import VAE
from gd.utils.ldos_transform import ldos_obs_from_linear

class RestartSampler:
    """
    Implements KPM validation and restart sampling.
    """
    def __init__(self, config: Dict[str, Any], kpm_forward: KPMForward, vae: VAE):
        self.config = config
        self.validation_cfg = config["validation"]
        self.kpm_cfg = self.validation_cfg["kpm_check"]
        self.restart_cfg = self.validation_cfg["restart"]
        self.kpm_forward = kpm_forward
        self.vae = vae

    def check(self, z: torch.Tensor, g_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        V_hat = self.vae.decode(z)
        g_pred = self._kpm_batch(V_hat)
        g_pred = ldos_obs_from_linear(g_pred, self.config)
        if g_pred.device != g_obs.device:
            g_pred = g_pred.to(g_obs.device)
        delta = self._delta(g_pred, g_obs)
        epsilon = self._epsilon(g_obs)
        restart_mask = delta > epsilon
        return delta, epsilon, restart_mask

    def add_restart_noise(self, z: torch.Tensor, diffusion_model: torch.nn.Module) -> torch.Tensor:
        t_restart = self.restart_cfg["t_restart"]
        noise_scale = self.restart_cfg["noise_scale"]
        alpha, sigma = diffusion_model.get_alpha_sigma(torch.tensor([t_restart], device=z.device))
        alpha = alpha.view(1, 1, 1, 1)
        sigma = sigma.view(1, 1, 1, 1)
        return alpha * z + sigma * noise_scale * torch.randn_like(z)

    def _kpm_batch(self, V_hat: torch.Tensor) -> torch.Tensor:
        B = V_hat.shape[0]
        g_list = []
        for i in range(B):
            g_list.append(self.kpm_forward.compute_ldos(V_hat[i, 0], self._energies()))
        return torch.stack(g_list, dim=0)

    def _energies(self) -> torch.Tensor:
        energies_cfg = self.config["data"]["energies"]
        if energies_cfg["mode"] == "linspace":
            return torch.linspace(energies_cfg["Emin"], energies_cfg["Emax"], self.config["data"]["K"]).tolist()
        return energies_cfg["list"]

    def _delta(self, g_pred: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((g_pred - g_obs) ** 2, dim=(1, 2, 3)))

    def _epsilon(self, g_obs: torch.Tensor) -> torch.Tensor:
        eps_cfg = self.kpm_cfg["epsilon"]
        if eps_cfg["mode"] == "absolute":
            return torch.full((g_obs.shape[0],), eps_cfg["absolute"], device=g_obs.device)
        sigma = self._sigma_noise(g_obs)
        return eps_cfg["c"] * sigma

    def _sigma_noise(self, g_obs: torch.Tensor) -> torch.Tensor:
        method = self.kpm_cfg["sigma_noise_est"]["method"]
        floor = self.kpm_cfg["sigma_noise_est"]["floor"]
        if method == "mad":
            med = torch.median(g_obs.view(g_obs.shape[0], -1), dim=1).values
            mad = torch.median(torch.abs(g_obs.view(g_obs.shape[0], -1) - med[:, None]), dim=1).values
            sigma = 1.4826 * mad
        else:
            sigma = g_obs.view(g_obs.shape[0], -1).std(dim=1)
        return torch.clamp(sigma, min=floor)

