from typing import Dict, Any, Optional, Union
import copy
import torch
from gd.models.vae import VAE
from gd.models.diffusion import LatentDiffusion
from gd.models.condition_encoder import ConditionEncoder
from gd.models.latent_green import LatentGreen
from gd.guidance.latent_guidance import LatentGuidance
from gd.guidance.restart import RestartSampler
from gd.utils.ldos_transform import force_linear_ldos_mode
from gd.utils.obs_layout import g_obs_to_model_view

class TeacherSampler:
    """
    High-quality sampling using the teacher diffusion model.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        diffusion_model: Optional[torch.nn.Module] = None,
        vae: Optional[torch.nn.Module] = None,
        condition_encoder: Optional[torch.nn.Module] = None,
        latent_green: Optional[torch.nn.Module] = None,
        guidance: Optional[LatentGuidance] = None,
    ):
        self.config = copy.deepcopy(config)
        force_linear_ldos_mode(self.config, verbose=False, context="teacher_sampler")
        self.diffusion = diffusion_model or LatentDiffusion(self.config)
        self.vae = vae or VAE(self.config)
        self.condition_encoder = condition_encoder or ConditionEncoder(self.config)
        self.latent_green = latent_green or LatentGreen(self.config)
        self.guidance = guidance or LatentGuidance(self.config)
        self.diff_cfg = self.config["diffusion"]
        self.guidance_cfg = self.config["guidance"]
        self.validation_cfg = self.config["validation"]
        self.restart = None
        if self.validation_cfg["enabled"] and self.validation_cfg["kpm_check"]["enabled"]:
            try:
                # Attempt to import KPMForward which relies on kwant
                from gd.data.kpm_forward import KPMForward
                __import__("kwant")  # Explicit dependency check without introducing an unused import.
                
                kpm = KPMForward(
                    {
                        "kpm": self.config["physics"]["kpm"],
                        "hamiltonian": self.config["physics"]["hamiltonian"],
                        "data": self.config.get("data", {}),
                        "rng_seed": self.config["project"]["seed"],
                    }
                )
                self.restart = RestartSampler(self.config, kpm, self.vae)
            except ImportError:
                print("Warning: 'kwant' not found. Disabling KPM check and Restart Guidance.")
                self.validation_cfg["kpm_check"]["enabled"] = False
                self.validation_cfg["restart"]["enabled"] = False
                self.restart = None
        self.unscale_factor: Union[float, torch.Tensor] = 1.0

    def sample(self, g_obs: torch.Tensor) -> torch.Tensor:
        """
        Generates a sample given a condition.
        Args:
            g_obs: Measurement tensor of shape (B, C, H, W) or canonical `(B,K,2,H,W)`.
        Returns:
            torch.Tensor: Reconstructed potential of shape (B, 1, H, W).
        """
        device = g_obs.device
        if g_obs.dim() == 5:
            g_obs = g_obs_to_model_view(g_obs, self.config.get("data", {}))
        B = g_obs.shape[0]
        H, W = g_obs.shape[-2:]
        C = self.config["vae"]["latent_channels"]
        h = H // self.config["vae"]["latent_downsample"]
        w = W // self.config["vae"]["latent_downsample"]
        z = torch.randn(B, C, h, w, device=device)
        
        T = self.diff_cfg["T"]
        steps = self.diff_cfg["sampler"]["steps"]
        eta = self.diff_cfg["sampler"]["eta"]
        # Use training-consistent timestep domain [0, T-1] and end sampling at t=0.
        if steps <= 1:
            timesteps = torch.tensor([max(T - 1, 0)], device=device, dtype=torch.long)
        else:
            timesteps = torch.linspace(max(T - 1, 0), 1, steps, device=device).long()
        z = self._sample_steps(z, g_obs, timesteps, eta, None)
        if self.restart is not None and self.validation_cfg["restart"]["enabled"]:
            max_restarts = self.validation_cfg["restart"]["max_restarts"]
            t_restart = self.validation_cfg["restart"]["t_restart"]
            for _ in range(max_restarts):
                with torch.no_grad():
                    delta, epsilon, mask = self.restart.check(z, g_obs)
                if not mask.any():
                    break
                z_restart = self.restart.add_restart_noise(z, self.diffusion)
                z = torch.where(mask[:, None, None, None], z_restart, z)
                steps_restart = max(2, int(steps * t_restart / T))
                restart_timesteps = torch.linspace(max(t_restart, 1), 1, steps_restart, device=device).long()
                z = self._sample_steps(z, g_obs, restart_timesteps, eta, mask)
        
        unscale = self.unscale_factor
        if isinstance(unscale, torch.Tensor):
            unscale = unscale.to(device=z.device, dtype=z.dtype)
            if unscale.numel() == 1:
                if abs(float(unscale.item()) - 1.0) > 1e-6:
                    z = z * unscale
            else:
                z = z * unscale
        else:
            if abs(float(unscale) - 1.0) > 1e-6:
                z = z * float(unscale)
            
        with torch.no_grad():
            return self.vae.decode(z)

    def _sample_steps(self, z: torch.Tensor, g_obs: torch.Tensor, timesteps: torch.Tensor, eta: float, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B = z.shape[0]
        for idx, t in enumerate(timesteps):
            t_batch = torch.full((B,), int(t.item()), device=z.device, dtype=torch.long)
            if idx + 1 < len(timesteps):
                t_prev_val = int(timesteps[idx + 1].item())
            else:
                t_prev_val = 0
            t_prev_batch = torch.full((B,), t_prev_val, device=z.device, dtype=torch.long)
            with torch.no_grad():
                z_new = self.diffusion.step(z, t_batch, g_obs, eta, t_prev=t_prev_batch)
            if self.guidance_cfg["enabled"] and self.guidance_cfg["use_latent_green"]:
                if t.item() <= self.guidance_cfg["lambda"]["start_step"]:
                    with torch.enable_grad():
                        z_new = self.guidance.apply(z_new, g_obs, t_batch, self.diffusion, self.latent_green)
            if mask is not None:
                z = torch.where(mask[:, None, None, None], z_new, z)
            else:
                z = z_new
        return z

