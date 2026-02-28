from typing import Dict, Any, List, Optional
import math
import torch
import os
import glob
import yaml
from bisect import bisect_right
from torch.utils.data import Dataset
from .potential_sampler import PotentialSampler
from .structural_defect_sampler import StructuralDefectSampler
from .kpm_forward import KPMForward
from .degradation import DegradationPipeline
from gd.utils.ldos_transform import force_linear_ldos_mode, ldos_transform_signature
from gd.utils.obs_layout import (
    cache_ldos_schema_metadata,
    expected_g_obs_shape,
    is_sublattice_resolved,
    require_graphene_if_sublattice_resolved,
    validate_canonical_g_obs,
)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

class GFDataset(Dataset):
    """
    Dataset for GF system.
    """
    def __init__(self, config: Dict[str, Any], split: str = "train"):
        self.config = config
        self.split = split
        self.data_cfg = config["data"]
        self.potential_cfg = config["potential_sampler"]
        self.physics_cfg = config["physics"]
        self.degradation_cfg = config["degradation"]
        self.project_cfg = config["project"]
        self.target_representation = str(self.data_cfg.get("target_representation", "ldos_ab"))
        self.structural_cfg = (
            self.potential_cfg.get("structural", {})
            if isinstance(self.potential_cfg.get("structural", {}), dict)
            else {}
        )
        self.structural_enabled = bool(self.structural_cfg.get("enabled", False))
        require_graphene_if_sublattice_resolved(config)
        self.sublattice_resolved_ldos = bool(is_sublattice_resolved(self.data_cfg))
        self.expected_g_obs_shape = expected_g_obs_shape(self.data_cfg, int(self.data_cfg["resolution"]))
        self.default_num_samples = int(self.data_cfg["split"][split] * self.data_cfg["num_samples_total"])
        self.num_samples = self.default_num_samples
        self.base_seed = self.project_cfg["seed"]
        self.ldos_cfg = self.data_cfg.get("ldos_transform", {"enabled": False})
        self.return_physics_meta = bool(self.data_cfg.get("return_physics_meta", False))
        t_cfg = self.physics_cfg.get("hamiltonian", {}).get("t", 1.0)
        self.hopping_is_random = isinstance(t_cfg, (list, tuple)) and len(t_cfg) > 1
        self.cache_require_physics_meta = bool(
            self.data_cfg.get("cache_require_physics_meta", False)
            or self.structural_enabled
            or self.hopping_is_random
        )
        self.attach_physics_meta = bool(self.return_physics_meta or self.cache_require_physics_meta)
        self.attach_defect_meta = bool(self.structural_enabled)
        self._physics_meta_cache_warned = False
        self._defect_meta_cache_warned = False
        self.cache_transform_signature = ldos_transform_signature(config)
        self.cache_require_transform_metadata = bool(
            self.data_cfg.get(
                "cache_require_transform_metadata",
                self.ldos_cfg.get("enabled", False) and self.ldos_cfg.get("apply_to_cache", True),
            )
        )
        # Physical LDOS should be non-negative. Keep this enabled by default and
        # allow opt-out for debugging legacy caches.
        self.enforce_nonnegative_ldos = bool(self.data_cfg.get("enforce_nonnegative_ldos", True))
        self.energies = self._build_energies()
        
        # Check for cache
        self.cache_dir = self.config.get("paths", {}).get("dataset_root", "data_cache")
        self.cache_path = os.path.join(self.cache_dir, f"{split}.pt")
        self.cache_index_path = os.path.join(self.cache_dir, f"{split}_index.yaml")
        self.cache_meta_path = os.path.join(self.cache_dir, f"{split}_meta.yaml")
        self.use_cache = False
        self.use_shards = False
        self.cached_data = None
        self.shard_files = []
        self.shard_cum_sizes = []
        self._current_shard_id = None
        self._current_shard_data = None
        self._shard_cache = {}
        self._shard_cache_order = []
        self._shard_cache_size = int(self.data_cfg.get("shard_cache_size", 1))
        self._cache_transform_signature_found = None
        self._cache_schema_meta_found: Dict[str, Any] = {}

        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, "r", encoding="utf-8") as f:
                index = yaml.safe_load(f) or {}
            self._cache_transform_signature_found = index.get("ldos_transform_signature")
            self._cache_schema_meta_found = {
                "ldos_schema_version": index.get("ldos_schema_version"),
                "target_representation": index.get("target_representation"),
                "contains_physics_meta": index.get("contains_physics_meta"),
                "contains_defect_meta": index.get("contains_defect_meta"),
                "potential_normalize": index.get("potential_normalize"),
                "sublattice_resolved_ldos": index.get("sublattice_resolved_ldos"),
                "ldos_canonical_layout": index.get("ldos_canonical_layout"),
                "ldos_model_layout": index.get("ldos_model_layout"),
                "sublattice_count": index.get("sublattice_count"),
            }
            shards = index.get("shards", [])
            self.shard_files = [os.path.join(self.cache_dir, s["file"]) for s in shards]
            sizes = [int(s["size"]) for s in shards]
            total = 0
            for s in sizes:
                total += s
                self.shard_cum_sizes.append(total)
            self.num_samples = total
            self.use_cache = True
            self.use_shards = True
            print(f"Using cached shards for {split}: {len(self.shard_files)} shards, {self.num_samples} samples from {self.cache_dir}")
        elif os.path.exists(self.cache_path):
            print(f"Loading {split} dataset from {self.cache_path}...")
            self.cached_data = torch.load(self.cache_path, weights_only=True)
            if os.path.exists(self.cache_meta_path):
                with open(self.cache_meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                self._cache_transform_signature_found = meta.get("ldos_transform_signature")
                self._cache_schema_meta_found = {
                    "ldos_schema_version": meta.get("ldos_schema_version"),
                    "target_representation": meta.get("target_representation"),
                    "contains_physics_meta": meta.get("contains_physics_meta"),
                    "contains_defect_meta": meta.get("contains_defect_meta"),
                    "potential_normalize": meta.get("potential_normalize"),
                    "sublattice_resolved_ldos": meta.get("sublattice_resolved_ldos"),
                    "ldos_canonical_layout": meta.get("ldos_canonical_layout"),
                    "ldos_model_layout": meta.get("ldos_model_layout"),
                    "sublattice_count": meta.get("sublattice_count"),
                }
            self.num_samples = len(self.cached_data)
            self.use_cache = True
        else:
            self.potential_sampler = PotentialSampler(self.potential_cfg)
            self.structural_sampler = StructuralDefectSampler(self.structural_cfg)
        if self.use_cache:
            cache_err: Optional[Exception] = None
            try:
                self._validate_cache_schema(split)
                if self.use_shards and self.shard_files:
                    shard_samples = torch.load(self.shard_files[0], weights_only=True)
                    sample = shard_samples[0] if shard_samples else None
                else:
                    sample = self.cached_data[0] if self.cached_data else None
                g_obs = sample.get("g_obs") if isinstance(sample, dict) else None
                validate_canonical_g_obs(
                    g_obs,
                    self.data_cfg,
                    resolution=int(self.data_cfg["resolution"]),
                    context=f"{split} cache g_obs",
                )
            except Exception as e:
                cache_err = e
            if cache_err is not None:
                if self.sublattice_resolved_ldos or self.attach_defect_meta or self.cache_require_physics_meta:
                    raise RuntimeError(
                        f"Incompatible cache for split={split!r} with current LDOS/physics schema. "
                        f"Expected g_obs shape {self.expected_g_obs_shape} and schema metadata "
                        f"{cache_ldos_schema_metadata(self.config)}. "
                        f"Detected metadata={self._cache_schema_meta_found}. "
                        f"Delete and rebuild cache in {self.cache_dir}."
                    ) from cache_err
                self.use_cache = False
                self.use_shards = False
                self.cached_data = None
                self.shard_files = []
                self.shard_cum_sizes = []
                self.num_samples = self.default_num_samples
                print(f"Cache mismatch for {split}; ignoring cache and regenerating on-the-fly.")
                self.potential_sampler = PotentialSampler(self.potential_cfg)
                self.structural_sampler = StructuralDefectSampler(self.structural_cfg)
            elif self.use_cache:
                cache_sig = self._cache_transform_signature_found
                if cache_sig is None:
                    if self.cache_require_transform_metadata:
                        self.use_cache = False
                        self.use_shards = False
                        self.cached_data = None
                        self.shard_files = []
                        self.shard_cum_sizes = []
                        self.num_samples = self.default_num_samples
                        print(
                            f"Cache transform metadata missing for {split}; refusing legacy cache to avoid LDOS scale/log mismatch. "
                            "Regenerating on-the-fly (rebuild cache once to restore fast loading)."
                        )
                        self.potential_sampler = PotentialSampler(self.potential_cfg)
                        self.structural_sampler = StructuralDefectSampler(self.structural_cfg)
                    else:
                        print(
                            f"Warning: cache transform metadata missing for {split}; cache will be used without LDOS transform verification."
                        )
                elif cache_sig != self.cache_transform_signature:
                    self.use_cache = False
                    self.use_shards = False
                    self.cached_data = None
                    self.shard_files = []
                    self.shard_cum_sizes = []
                    self.num_samples = self.default_num_samples
                    print(
                        f"Cache LDOS transform signature mismatch for {split}; ignoring cache.\n"
                        f"  cache:   {cache_sig}\n"
                        f"  current: {self.cache_transform_signature}"
                    )
                    self.potential_sampler = PotentialSampler(self.potential_cfg)
                    self.structural_sampler = StructuralDefectSampler(self.structural_cfg)
        if not hasattr(self, "potential_sampler"):
            self.potential_sampler = PotentialSampler(self.potential_cfg)
        if not hasattr(self, "structural_sampler"):
            self.structural_sampler = StructuralDefectSampler(self.structural_cfg)
        # Always prepare physics forward and degradation for fallback/recompute
        try:
            self.kpm = KPMForward(
                {
                    "kpm": self.physics_cfg["kpm"],
                    "hamiltonian": self.physics_cfg["hamiltonian"],
                    "data": self.data_cfg,
                    "rng_seed": self.base_seed,
                }
            )
        except Exception as e:
            self.kpm = None
        self.degradation = DegradationPipeline(self.degradation_cfg)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing:
                - 'V': (H, W)
                - 'g_obs': (K, H, W)
        """
        if self.use_shards:
            shard_id = bisect_right(self.shard_cum_sizes, idx)
            start = 0 if shard_id == 0 else self.shard_cum_sizes[shard_id - 1]
            local_idx = idx - start
            if self._current_shard_id != shard_id:
                cached = self._shard_cache.get(shard_id)
                if cached is None:
                    data = torch.load(self.shard_files[shard_id], weights_only=True)
                    if self._shard_cache_size > 0:
                        self._shard_cache[shard_id] = data
                        self._shard_cache_order.append(shard_id)
                        if len(self._shard_cache_order) > self._shard_cache_size:
                            evict_id = self._shard_cache_order.pop(0)
                            self._shard_cache.pop(evict_id, None)
                    cached = data
                self._current_shard_data = cached
                self._current_shard_id = shard_id
            sample = self._current_shard_data[local_idx]
            if self.ldos_cfg.get("enabled", False) and self.ldos_cfg.get("apply_to_cache", True):
                cache_scaled = bool(self.ldos_cfg.get("cache_scaled", False))
                if not cache_scaled and self._should_transform_cache(sample["g_obs"]):
                    sample = dict(sample)
                    sample["g_obs"] = self._transform_ldos(sample["g_obs"])
            sample = self._postprocess_cached_sample(sample)
            sample = self._maybe_attach_cached_physics_meta(sample)
            sample = self._maybe_attach_cached_defect_meta(sample)
            return sample
        if self.use_cache:
            sample = self.cached_data[idx]
            if self.ldos_cfg.get("enabled", False) and self.ldos_cfg.get("apply_to_cache", True):
                cache_scaled = bool(self.ldos_cfg.get("cache_scaled", False))
                if not cache_scaled and self._should_transform_cache(sample["g_obs"]):
                    sample = dict(sample)
                    sample["g_obs"] = self._transform_ldos(sample["g_obs"])
            sample = self._postprocess_cached_sample(sample)
            sample = self._maybe_attach_cached_physics_meta(sample)
            sample = self._maybe_attach_cached_defect_meta(sample)
            return sample
            
        H = self.data_cfg["resolution"]
        W = self.data_cfg["resolution"]
        V = self.potential_sampler.sample(H, W, family=None, seed=self.base_seed + idx)
        defect_meta = None
        if self.structural_enabled:
            lattice_type_cfg = str(self.physics_cfg.get("hamiltonian", {}).get("type", "square_lattice")).lower()
            if lattice_type_cfg in ("graphene", "honeycomb", "random"):
                defect_meta = self.structural_sampler.sample_graphene(H, W, seed=self.base_seed + idx)
        g_ideal = self.kpm.compute_ldos(V, self.energies, defect_meta=defect_meta)
        g_ideal = self._clamp_nonnegative_ldos(g_ideal)
        g_obs = self.degradation(g_ideal)
        g_obs = self._clamp_nonnegative_ldos(g_obs)
        if self.ldos_cfg.get("enabled", False):
            g_obs = self._transform_ldos(g_obs)
            # In linear observation mode (default), keep transformed cache non-negative too.
            if not self._obs_log_transform_enabled():
                g_obs = self._clamp_nonnegative_ldos(g_obs)
        sample = {"V": V, "g_obs": g_obs}
        if self.attach_defect_meta and defect_meta is not None:
            sample["defect_meta"] = defect_meta
        if self.attach_physics_meta and self.kpm is not None:
            meta = getattr(self.kpm, "get_last_sample_meta", lambda: {})() or {}
            lattice_type = str(meta.get("lattice_type", "unknown"))
            lattice_type_id = 1 if lattice_type == "graphene" else (0 if lattice_type == "square_lattice" else -1)
            sample["physics_meta"] = {
                "lattice_type_id": torch.tensor(lattice_type_id, dtype=torch.long),
                "hopping": torch.tensor(float(meta.get("hopping", float("nan"))), dtype=torch.float32),
                "eta": torch.tensor(float(meta.get("eta", self.physics_cfg.get("kpm", {}).get("eta", 0.01))), dtype=torch.float32),
                "target_type_id": torch.tensor(0 if self.target_representation == "ldos_ab" else 1, dtype=torch.long),
                "num_orbs": torch.tensor(int(meta.get("num_orbs", 1)), dtype=torch.long),
                "graphene_simplified_branch": torch.tensor(
                    1 if bool(meta.get("graphene_simplified_branch", False)) else 0, dtype=torch.long
                ),
            }
        return sample

    def _build_energies(self) -> List[float]:
        energies_cfg = self.data_cfg["energies"]
        if energies_cfg["mode"] == "linspace":
            return torch.linspace(energies_cfg["Emin"], energies_cfg["Emax"], self.data_cfg["K"]).tolist()
        return energies_cfg["list"]

    def _transform_ldos(self, g_obs: torch.Tensor) -> torch.Tensor:
        x = g_obs
        log_cfg = self.ldos_cfg.get("log", {})
        if log_cfg.get("enabled", False):
            eps = log_cfg.get("eps", 1.0e-6)
            x = torch.log(x.clamp_min(0) + eps)
        quant_cfg = self.ldos_cfg.get("quantile", {})
        if quant_cfg.get("enabled", False):
            q_eps = quant_cfg.get("eps", 1.0e-6)
            x = self._quantile_gaussianize(x, q_eps)
        scale = self.ldos_cfg.get("scale", 1.0)
        if scale is not None and float(scale) != 1.0:
            x = x * float(scale)
        return x

    def _obs_log_transform_enabled(self) -> bool:
        if not bool(self.ldos_cfg.get("enabled", False)):
            return False
        log_cfg = self.ldos_cfg.get("log", {})
        return bool(isinstance(log_cfg, dict) and log_cfg.get("enabled", False))

    def _clamp_nonnegative_ldos(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enforce_nonnegative_ldos:
            return x
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            return x
        if x.numel() == 0:
            return x
        if float(torch.min(x).item()) >= 0.0:
            return x
        return x.clamp_min(0.0)

    def _postprocess_cached_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enforce_nonnegative_ldos:
            return sample
        if not isinstance(sample, dict) or "g_obs" not in sample:
            return sample
        if self._obs_log_transform_enabled():
            return sample
        g_obs = sample["g_obs"]
        g_fixed = self._clamp_nonnegative_ldos(g_obs)
        if g_fixed is g_obs:
            return sample
        out = dict(sample)
        out["g_obs"] = g_fixed
        return out

    def _maybe_attach_cached_physics_meta(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.attach_physics_meta:
            return sample
        if isinstance(sample, dict) and "physics_meta" in sample:
            return sample
        if not self._physics_meta_cache_warned:
            print(
                "Warning: physics metadata is required/enabled but cached samples do not contain 'physics_meta'. "
                "Rebuild dataset cache if you need lattice/material metadata for mixed-material training."
            )
            self._physics_meta_cache_warned = True
        return sample

    def _maybe_attach_cached_defect_meta(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.attach_defect_meta:
            return sample
        if isinstance(sample, dict) and "defect_meta" in sample:
            return sample
        if not self._defect_meta_cache_warned:
            print(
                "Warning: structural defects are enabled but cached samples do not contain 'defect_meta'. "
                "Rebuild dataset cache to enable graphene structural-defect physics training."
            )
            self._defect_meta_cache_warned = True
        return sample

    def _should_transform_cache(self, g_obs: torch.Tensor) -> bool:
        if g_obs.numel() == 0:
            return True
        if not torch.isfinite(g_obs).all():
            return False
        return True

    def _validate_cache_schema(self, split: str) -> None:
        expected = cache_ldos_schema_metadata(self.config)
        if int(expected.get("ldos_schema_version", 1)) < 3 and not self.sublattice_resolved_ldos:
            return
        meta = self._cache_schema_meta_found or {}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"cache schema mismatch for {split}: key={key!r}, expected={value!r}, found={meta.get(key)!r}"
                )
        if self.cache_require_physics_meta:
            if not bool(meta.get("contains_physics_meta", False)):
                raise ValueError(f"cache schema mismatch for {split}: physics_meta required but cache metadata says missing")
        if self.attach_defect_meta:
            if not bool(meta.get("contains_defect_meta", False)):
                raise ValueError(f"cache schema mismatch for {split}: defect_meta required but cache metadata says missing")

    def _is_degenerate(self, g_obs: torch.Tensor) -> bool:
        if g_obs.numel() == 0:
            return True
        if not torch.isfinite(g_obs).all():
            return True
        rng = g_obs.max().item() - g_obs.min().item()
        return rng < 1.0e-5

    def _quantile_gaussianize(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        if x.dim() == 3:
            out = torch.empty_like(x)
            for i in range(x.shape[0]):
                out[i] = self._gaussianize_2d(x[i], eps)
            return out
        if x.dim() == 4:
            out = torch.empty_like(x)
            for b in range(x.shape[0]):
                for i in range(x.shape[1]):
                    out[b, i] = self._gaussianize_2d(x[b, i], eps)
            return out
        if x.dim() == 5:
            out = torch.empty_like(x)
            for b in range(x.shape[0]):
                for i in range(x.shape[1]):
                    for s in range(x.shape[2]):
                        out[b, i, s] = self._gaussianize_2d(x[b, i, s], eps)
            return out
        return x

    def _gaussianize_2d(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        flat = x.flatten()
        order = torch.argsort(flat)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(flat.numel(), device=flat.device)
        u = (ranks.float() + 0.5) / float(flat.numel())
        u = u.clamp(eps, 1.0 - eps)
        z = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
        return z.view_as(x)


def _v_only_cache_paths(cache_dir: str, split: str) -> Dict[str, str]:
    return {
        "single": os.path.join(cache_dir, f"{split}_V_only.pt"),
        "index": os.path.join(cache_dir, f"{split}_V_only_index.yaml"),
    }


def _v_only_shard_name(src_shard_name: str) -> str:
    if "_shard_" in src_shard_name:
        return src_shard_name.replace("_shard_", "_V_only_shard_", 1)
    base, ext = os.path.splitext(src_shard_name)
    return f"{base}_V_only{ext}"


def _unbatch_sample_tree(x):
    if isinstance(x, torch.Tensor):
        return x[0]
    if isinstance(x, dict):
        return {k: _unbatch_sample_tree(v) for k, v in x.items()}
    return x


def ensure_v_only_cache(config: Dict[str, Any], split: str = "train", verbose: bool = True) -> None:
    """
    Build a V-only cache sidecar from existing cached samples (V + g_obs) so VAE training
    can avoid deserializing g_obs on every batch.
    """
    cache_dir = config.get("paths", {}).get("dataset_root", "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    source_index_path = os.path.join(cache_dir, f"{split}_index.yaml")
    source_cache_path = os.path.join(cache_dir, f"{split}.pt")
    source_meta_path = os.path.join(cache_dir, f"{split}_meta.yaml")
    v_paths = _v_only_cache_paths(cache_dir, split)

    if os.path.exists(source_index_path):
        with open(source_index_path, "r", encoding="utf-8") as f:
            src_index = yaml.safe_load(f) or {}
        src_shards = src_index.get("shards", [])
        if not src_shards:
            return
        v_index_exists = os.path.exists(v_paths["index"])
        all_v_shards_exist = True
        for shard in src_shards:
            shard_name = str(shard["file"])
            v_shard_path = os.path.join(cache_dir, _v_only_shard_name(shard_name))
            if not os.path.exists(v_shard_path):
                all_v_shards_exist = False
                break
        if v_index_exists and all_v_shards_exist:
            return

        if verbose:
            print(f"Preparing V-only cache for {split} from shard cache in {cache_dir}...")
        v_index = {"source_index": os.path.basename(source_index_path), "shards": []}
        for shard in tqdm(src_shards, desc=f"Extracting V-only {split}"):
            shard_name = str(shard["file"])
            src_shard_path = os.path.join(cache_dir, shard_name)
            v_shard_name = _v_only_shard_name(shard_name)
            v_shard_path = os.path.join(cache_dir, v_shard_name)
            if not os.path.exists(v_shard_path):
                samples = torch.load(src_shard_path, weights_only=True)
                v_list = [s["V"] for s in samples]
                if len(v_list) > 0 and all(isinstance(v, torch.Tensor) and v.shape == v_list[0].shape for v in v_list):
                    v_payload = torch.stack(v_list, dim=0)
                else:
                    v_payload = v_list
                torch.save(v_payload, v_shard_path)
            v_index["shards"].append({"file": v_shard_name, "size": int(shard["size"])})
        with open(v_paths["index"], "w", encoding="utf-8") as f:
            yaml.safe_dump(v_index, f, allow_unicode=True, sort_keys=False)
        return

    if os.path.exists(source_cache_path):
        if os.path.exists(v_paths["single"]):
            return
        if verbose:
            print(f"Preparing V-only cache for {split} from {source_cache_path}...")
        samples = torch.load(source_cache_path, weights_only=True)
        v_list = [s["V"] for s in samples]
        if len(v_list) > 0 and all(isinstance(v, torch.Tensor) and v.shape == v_list[0].shape for v in v_list):
            v_payload = torch.stack(v_list, dim=0)
        else:
            v_payload = v_list
        torch.save(v_payload, v_paths["single"])
        meta = {}
        if os.path.exists(source_meta_path):
            with open(source_meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
        meta = {"source_meta": os.path.basename(source_meta_path), **meta}
        with open(v_paths["index"], "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)


class GFVOnlyDataset(Dataset):
    """
    VAE-focused dataset that reads a V-only sidecar cache when available to avoid loading g_obs.
    Falls back to GFDataset and strips keys if no cache exists.
    """
    def __init__(self, config: Dict[str, Any], split: str = "train"):
        self.config = config
        self.split = split
        self.data_cfg = config["data"]
        self.cache_dir = self.config.get("paths", {}).get("dataset_root", "data_cache")
        self.cache_path = os.path.join(self.cache_dir, f"{split}.pt")
        self.cache_index_path = os.path.join(self.cache_dir, f"{split}_index.yaml")
        self.v_paths = _v_only_cache_paths(self.cache_dir, split)
        self.use_v_only_cache = False
        self.use_shards = False
        self.num_samples = int(self.data_cfg["split"][split] * self.data_cfg["num_samples_total"])
        self.v_cached_data = None
        self.v_shard_files: List[str] = []
        self.shard_cum_sizes: List[int] = []
        self._current_shard_id: Optional[int] = None
        self._current_shard_data = None
        self._shard_cache = {}
        self._shard_cache_order: List[int] = []
        self._shard_cache_size = int(self.data_cfg.get("shard_cache_size", 1))
        self._fallback_dataset: Optional[GFDataset] = None

        if os.path.exists(self.v_paths["index"]) and os.path.exists(self.cache_index_path):
            with open(self.v_paths["index"], "r", encoding="utf-8") as f:
                v_index = yaml.safe_load(f) or {}
            shards = v_index.get("shards", [])
            if shards:
                total = 0
                for s in shards:
                    self.v_shard_files.append(os.path.join(self.cache_dir, str(s["file"])))
                    total += int(s["size"])
                    self.shard_cum_sizes.append(total)
                if self.v_shard_files and all(os.path.exists(p) for p in self.v_shard_files):
                    self.num_samples = total
                    self.use_v_only_cache = True
                    self.use_shards = True
                    print(f"Using V-only cached shards for {split}: {len(self.v_shard_files)} shards, {self.num_samples} samples from {self.cache_dir}")

        if (not self.use_v_only_cache) and os.path.exists(self.v_paths["single"]):
            print(f"Loading V-only {split} dataset from {self.v_paths['single']}...")
            self.v_cached_data = torch.load(self.v_paths["single"], weights_only=True)
            self.num_samples = len(self.v_cached_data)
            self.use_v_only_cache = True
            self.use_shards = False

        if not self.use_v_only_cache:
            self._fallback_dataset = GFDataset(config, split=split)
            self.num_samples = len(self._fallback_dataset)

    def __len__(self) -> int:
        return self.num_samples

    def _wrap_v(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"V": v}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.use_shards:
            shard_id = bisect_right(self.shard_cum_sizes, idx)
            start = 0 if shard_id == 0 else self.shard_cum_sizes[shard_id - 1]
            local_idx = idx - start
            if self._current_shard_id != shard_id:
                cached = self._shard_cache.get(shard_id)
                if cached is None:
                    data = torch.load(self.v_shard_files[shard_id], weights_only=True)
                    if self._shard_cache_size > 0:
                        self._shard_cache[shard_id] = data
                        self._shard_cache_order.append(shard_id)
                        if len(self._shard_cache_order) > self._shard_cache_size:
                            evict_id = self._shard_cache_order.pop(0)
                            self._shard_cache.pop(evict_id, None)
                    cached = data
                self._current_shard_data = cached
                self._current_shard_id = shard_id
            shard_data = self._current_shard_data
            return self._wrap_v(shard_data[local_idx])

        if self.use_v_only_cache:
            return self._wrap_v(self.v_cached_data[idx])

        sample = self._fallback_dataset[idx]
        return self._wrap_v(sample["V"])

def generate_cache(config: Dict[str, Any], splits: List[str]) -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    force_linear_ldos_mode(config, verbose=True, context="generate_cache")
    require_graphene_if_sublattice_resolved(config)
    cache_dir = config.get("paths", {}).get("dataset_root", "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Data generation should run on CPU, but must not leak that setting into later
    # pipeline stages (e.g. VAE training) when the same config object is reused.
    original_device = config.get("project", {}).get("device", "cpu")
    config["project"]["device"] = "cpu"
    shard_size = int(config.get("data", {}).get("cache_shard_size", 0) or 0)
    num_workers = config.get("data", {}).get("num_workers", 4)
    import multiprocessing
    available_cpu = multiprocessing.cpu_count()
    if num_workers > available_cpu:
        num_workers = available_cpu
    print(f"Using {num_workers} worker processes for data generation.")
    from torch.utils.data import DataLoader
    try:
        for split in splits:
            print(f"Generating {split} set...")
            cache_path = os.path.join(cache_dir, f"{split}.pt")
            index_path = os.path.join(cache_dir, f"{split}_index.yaml")
            meta_path = os.path.join(cache_dir, f"{split}_meta.yaml")
            if shard_size > 0:
                for f in glob.glob(os.path.join(cache_dir, f"{split}_shard_*.pt")):
                    os.remove(f)
                for f in glob.glob(os.path.join(cache_dir, f"{split}_V_only_shard_*.pt")):
                    os.remove(f)
                if os.path.exists(index_path):
                    os.remove(index_path)
                v_only_index = os.path.join(cache_dir, f"{split}_V_only_index.yaml")
                if os.path.exists(v_only_index):
                    os.remove(v_only_index)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            elif os.path.exists(cache_path):
                print(f"  Warning: {cache_path} already exists. Overwriting...")
                os.remove(cache_path)
                v_only_single = os.path.join(cache_dir, f"{split}_V_only.pt")
                v_only_index = os.path.join(cache_dir, f"{split}_V_only_index.yaml")
                if os.path.exists(v_only_single):
                    os.remove(v_only_single)
                if os.path.exists(v_only_index):
                    os.remove(v_only_index)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            dataset = GFDataset(config, split=split)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False
            )
            if shard_size > 0:
                index = {
                    "ldos_transform_signature": ldos_transform_signature(config),
                    **cache_ldos_schema_metadata(config),
                    "shards": [],
                }
                shard_samples = []
                shard_id = 0
                for batch in tqdm(loader, desc=f"Generating {split}"):
                    sample = _unbatch_sample_tree(batch)
                    shard_samples.append(sample)
                    if len(shard_samples) >= shard_size:
                        shard_file = f"{split}_shard_{shard_id:04d}.pt"
                        torch.save(shard_samples, os.path.join(cache_dir, shard_file))
                        index["shards"].append({"file": shard_file, "size": len(shard_samples)})
                        shard_samples = []
                        shard_id += 1
                if shard_samples:
                    shard_file = f"{split}_shard_{shard_id:04d}.pt"
                    torch.save(shard_samples, os.path.join(cache_dir, shard_file))
                    index["shards"].append({"file": shard_file, "size": len(shard_samples)})
                with open(index_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(index, f, allow_unicode=True, sort_keys=False)
                print(f"  Saved {len(index['shards'])} shards to {cache_dir}...")
            else:
                samples = []
                for batch in tqdm(loader, desc=f"Generating {split}"):
                    sample = _unbatch_sample_tree(batch)
                    samples.append(sample)
                print(f"  Saving {len(samples)} samples to {cache_path}...")
                torch.save(samples, cache_path)
                meta = {"ldos_transform_signature": ldos_transform_signature(config), **cache_ldos_schema_metadata(config)}
                with open(meta_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
            print("  Done.")
    finally:
        config["project"]["device"] = original_device

if __name__ == "__main__":
    import sys
    import os
    from gd.utils.config_utils import load_config

    # Fix for OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Try to locate config
    config_path = "gd/configs/default.yaml"
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print("Error: Could not find config file.")
        sys.exit(1)
        
    generate_cache(config, ["train", "val"])

