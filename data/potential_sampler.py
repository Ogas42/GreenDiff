from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

class PotentialSampler:
    """
    势能采样器：生成用于量子散射模拟的2D势能景观 V(r)。
    支持点缺陷、簇状缺陷、相关噪声、畴壁以及安德森无序等多种物理场景。
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.family_default = config["family"]
        self.normalize = config.get("normalize", False)
        self.mixed_cfg = config.get("mixed")
        self.point_cfg = config.get("point_impurity")
        self.cluster_cfg = config.get("clustered")
        self.corr_cfg = config.get("correlated_noise")
        self.wall_cfg = config.get("domain_wall")
        self.anderson_cfg = config.get("anderson", {"amplitude_range": [-1.0, 1.0]})
        self.blur_kernel_truncate = float(config.get("blur_kernel_truncate", 4.0))

    def sample(self, H: int, W: int, family: Optional[str], seed: Optional[int]) -> torch.Tensor:
        """
        采样一个势能样本。

        参数:
            H: 高度 (格点数)
            W: 宽度 (格点数)
            family: 缺陷族名称 (可选，默认为配置中的 family)
            seed: 随机种子 (可选)
        返回:
            torch.Tensor: 形状为 (H, W) 的势能张量
        """
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        if family is None:
            family = self.family_default
        V = self._sample_family(H, W, family, gen)
        return self._normalize(V) if self.normalize else V

    def _sample_family(self, H: int, W: int, family: str, gen: torch.Generator) -> torch.Tensor:
        if family == "point_impurity":
            return self._point_impurity(H, W, gen)
        if family == "clustered":
            return self._clustered(H, W, gen)
        if family == "correlated_noise":
            return self._correlated_noise(H, W, gen)
        if family == "domain_wall":
            return self._domain_wall(H, W, gen)
        if family == "anderson":
            return self._anderson(H, W, gen)
        if family == "mixed":
            return self._mixed(H, W, gen)
        raise ValueError(f"Unknown family: {family}")

    def _point_impurity(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成点缺陷势能 (稀疏分布的杂质)"""
        if self.point_cfg is None:
            raise KeyError("potential_sampler.point_impurity config is required for family 'point_impurity'")
        V = torch.zeros(H, W)
        n_min, n_max = self.point_cfg["num_points_range"]
        amp_min, amp_max = self.point_cfg["amplitude_range"]
        sigma_min, sigma_max = self.point_cfg["blob_sigma_range"]
        num_points = torch.randint(n_min, n_max + 1, (1,), generator=gen).item()
        xs = torch.randint(0, H, (num_points,), generator=gen)
        ys = torch.randint(0, W, (num_points,), generator=gen)
        amps = amp_min + (amp_max - amp_min) * torch.rand(num_points, generator=gen)
        V[xs, ys] = amps
        sigma = sigma_min + (sigma_max - sigma_min) * torch.rand(1, generator=gen).item()
        if sigma > 0:
            V = self._gaussian_blur_2d(V, sigma)
        return V

    def _clustered(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成簇状缺陷势能 (团簇状的势垒或势阱)"""
        if self.cluster_cfg is None:
            raise KeyError("potential_sampler.clustered config is required for family 'clustered'")
        V = torch.zeros(H, W)
        n_clusters_min, n_clusters_max = self.cluster_cfg["num_clusters_range"]
        radius_min, radius_max = self.cluster_cfg["radius_range"]
        amp_min, amp_max = self.cluster_cfg["amplitude_range"]
        num_clusters = torch.randint(n_clusters_min, n_clusters_max + 1, (1,), generator=gen).item()
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        for _ in range(num_clusters):
            cx = torch.randint(0, H, (1,), generator=gen).item()
            cy = torch.randint(0, W, (1,), generator=gen).item()
            radius = radius_min + (radius_max - radius_min) * torch.rand(1, generator=gen).item()
            amp = amp_min + (amp_max - amp_min) * torch.rand(1, generator=gen).item()
            mask = ((xs - cy) ** 2 + (ys - cx) ** 2) <= radius**2
            V = V + amp * mask.float()
        return V

    def _correlated_noise(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成具有空间相关性的随机势能 (模拟背景无序)"""
        if self.corr_cfg is None:
            raise KeyError("potential_sampler.correlated_noise config is required for family 'correlated_noise'")
        sigma_min, sigma_max = self.corr_cfg["corr_length_range"]
        amp_min, amp_max = self.corr_cfg["amplitude_range"]
        corr_length = sigma_min + (sigma_max - sigma_min) * torch.rand(1, generator=gen).item()
        sigma = 1.0 / corr_length
        noise = torch.randn(H, W, generator=gen)
        fy = torch.fft.fftfreq(H)
        fx = torch.fft.rfftfreq(W)
        ky, kx = torch.meshgrid(fy, fx, indexing="ij")
        radius2 = kx**2 + ky**2
        filt = torch.exp(-radius2 / (2.0 * sigma**2))
        spectrum = torch.fft.rfft2(noise)
        filtered = torch.fft.irfft2(spectrum * filt, s=(H, W))
        amp = amp_min + (amp_max - amp_min) * torch.rand(1, generator=gen).item()
        return filtered * amp

    def _domain_wall(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成畴壁势能 (模拟材料中的晶界或相位分离)"""
        if self.wall_cfg is None:
            raise KeyError("potential_sampler.domain_wall config is required for family 'domain_wall'")
        n_min, n_max = self.wall_cfg["num_regions_range"]
        amp_min, amp_max = self.wall_cfg["amplitude_range"]
        smooth = self.wall_cfg["smooth_boundary"]
        smooth_sigma = self.wall_cfg["boundary_smooth_sigma"]
        num_regions = torch.randint(n_min, n_max + 1, (1,), generator=gen).item()
        field = torch.randn(H, W, generator=gen)
        if smooth:
            field = self._gaussian_blur_2d(field, smooth_sigma)
        quantiles = torch.linspace(0, 1, num_regions + 1)[1:-1]
        thresholds = torch.quantile(field, quantiles)
        labels = torch.zeros_like(field, dtype=torch.long)
        for i, t in enumerate(thresholds):
            labels = labels + (field > t).long()
        amps = amp_min + (amp_max - amp_min) * torch.rand(num_regions, generator=gen)
        V = torch.zeros_like(field)
        for i in range(num_regions):
            V = V + (labels == i).float() * amps[i]
        return V

    def _mixed(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成混合缺陷势能 (多种缺陷类型的加权组合)"""
        if self.mixed_cfg is None:
            raise KeyError("potential_sampler.mixed config is required for family 'mixed'")
        weights_dict = self.mixed_cfg["weights"]
        families = list(weights_dict.keys())
        weights = torch.tensor(list(weights_dict.values()), dtype=torch.float32)
        weights = weights / weights.sum()
        V = torch.zeros(H, W)
        for family, w in zip(families, weights):
            V = V + w * self._sample_family(H, W, family, gen)
        return V
    
    def _anderson(self, H: int, W: int, gen: torch.Generator) -> torch.Tensor:
        """生成安德森无序势能 (格点上的随机不均匀性)"""
        amp_min, amp_max = self.anderson_cfg["amplitude_range"]
        return amp_min + (amp_max - amp_min) * torch.rand(H, W, generator=gen)

    def _normalize(self, V: torch.Tensor) -> torch.Tensor:
        """将势能张量归一化到 [-1, 1] 区间"""
        vmin = V.min()
        vmax = V.max()
        if torch.isclose(vmin, vmax):
            return torch.zeros_like(V)
        return 2.0 * (V - vmin) / (vmax - vmin) - 1.0

    def _gaussian_blur_2d(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """对2D张量应用高斯模糊 (用于平滑缺陷边缘)"""
        radius = int(self.blur_kernel_truncate * sigma)
        if radius <= 0:
            return x
        size = 2 * radius + 1
        coords = torch.arange(size) - radius
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        weight = kernel.view(1, 1, size, size)
        x4 = x.view(1, 1, x.shape[0], x.shape[1])
        out = F.conv2d(x4, weight, padding=radius)
        return out.view_as(x)
