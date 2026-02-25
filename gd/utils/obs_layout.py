from typing import Any, Dict, Mapping

import torch


_CANONICAL_LAYOUT = "k_s_h_w"
_MODEL_LAYOUT = "flat_channels"


def _as_data_cfg(config_or_data_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "data" in config_or_data_cfg and isinstance(config_or_data_cfg.get("data"), Mapping):
        return config_or_data_cfg["data"]
    return config_or_data_cfg


def is_sublattice_resolved(config_or_data_cfg: Mapping[str, Any]) -> bool:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    return bool(data_cfg.get("sublattice_resolved_ldos", False))


def obs_sublattice_count(config_or_data_cfg: Mapping[str, Any]) -> int:
    return 2 if is_sublattice_resolved(config_or_data_cfg) else 1


def obs_channel_count(config_or_data_cfg: Mapping[str, Any]) -> int:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    k = int(data_cfg["K"])
    return k * obs_sublattice_count(data_cfg)


def _require_layout_flags(data_cfg: Mapping[str, Any]) -> None:
    canonical = str(data_cfg.get("ldos_canonical_layout", _CANONICAL_LAYOUT))
    model = str(data_cfg.get("ldos_model_layout", _MODEL_LAYOUT))
    if canonical != _CANONICAL_LAYOUT:
        raise ValueError(
            f"Unsupported LDOS canonical layout {canonical!r}; expected {_CANONICAL_LAYOUT!r}."
        )
    if model != _MODEL_LAYOUT:
        raise ValueError(
            f"Unsupported LDOS model layout {model!r}; expected {_MODEL_LAYOUT!r}."
        )


def _ensure_canonical_rank(x: torch.Tensor, *, allow_batch: bool = True) -> None:
    if x.dim() == 4:
        return
    if allow_batch and x.dim() == 5:
        return
    raise ValueError(f"Expected canonical LDOS rank 4 or 5, got shape {tuple(x.shape)}")


def _ensure_model_rank(x: torch.Tensor) -> None:
    if x.dim() in (3, 4):
        return
    raise ValueError(f"Expected model-view LDOS rank 3 or 4, got shape {tuple(x.shape)}")


def g_obs_to_model_view(x: torch.Tensor, config_or_data_cfg: Mapping[str, Any]) -> torch.Tensor:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    if not is_sublattice_resolved(data_cfg):
        return x
    _require_layout_flags(data_cfg)
    _ensure_canonical_rank(x)
    k = int(data_cfg["K"])
    if x.dim() == 4:
        if x.shape[0] != k or x.shape[1] != 2:
            raise ValueError(f"Expected canonical shape (K,2,H,W) with K={k}, got {tuple(x.shape)}")
        return x.reshape(k * 2, x.shape[2], x.shape[3])
    if x.shape[1] != k or x.shape[2] != 2:
        raise ValueError(f"Expected canonical batched shape (B,K,2,H,W) with K={k}, got {tuple(x.shape)}")
    return x.reshape(x.shape[0], k * 2, x.shape[3], x.shape[4])


def g_obs_to_canonical_view(x: torch.Tensor, config_or_data_cfg: Mapping[str, Any]) -> torch.Tensor:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    if not is_sublattice_resolved(data_cfg):
        return x
    _require_layout_flags(data_cfg)
    _ensure_model_rank(x)
    k = int(data_cfg["K"])
    c_expected = 2 * k
    if x.dim() == 3:
        if x.shape[0] != c_expected:
            raise ValueError(f"Expected model-view shape (2K,H,W) with 2K={c_expected}, got {tuple(x.shape)}")
        return x.reshape(k, 2, x.shape[1], x.shape[2])
    if x.shape[1] != c_expected:
        raise ValueError(f"Expected model-view shape (B,2K,H,W) with 2K={c_expected}, got {tuple(x.shape)}")
    return x.reshape(x.shape[0], k, 2, x.shape[2], x.shape[3])


def flatten_sub_for_energy_ops(x_canonical: torch.Tensor) -> torch.Tensor:
    _ensure_canonical_rank(x_canonical)
    if x_canonical.dim() == 4:
        x = x_canonical.unsqueeze(0)
        squeeze = True
    else:
        x = x_canonical
        squeeze = False
    # (B, K, 2, H, W) -> (B, 2, K, H, W) -> (B*2, K, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    return x.squeeze(0) if squeeze else x


def unflatten_sub_from_energy_ops(x_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x_flat.dim() != 4:
        raise ValueError(f"Expected flat energy-ops tensor rank 4, got {tuple(x_flat.shape)}")
    if x_flat.shape[0] != batch_size * 2:
        raise ValueError(
            f"Expected flat batch dimension {batch_size * 2}, got {x_flat.shape[0]}"
        )
    x = x_flat.reshape(batch_size, 2, x_flat.shape[1], x_flat.shape[2], x_flat.shape[3])
    return x.permute(0, 2, 1, 3, 4).contiguous()


def aggregate_sublattice_ldos(x_canonical: torch.Tensor) -> torch.Tensor:
    _ensure_canonical_rank(x_canonical)
    if x_canonical.dim() == 4:
        return x_canonical.sum(dim=1)
    return x_canonical.sum(dim=2)


def expected_g_obs_shape(config_or_data_cfg: Mapping[str, Any], resolution: int) -> tuple:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    k = int(data_cfg["K"])
    if is_sublattice_resolved(data_cfg):
        _require_layout_flags(data_cfg)
        return (k, 2, resolution, resolution)
    return (k, resolution, resolution)


def validate_canonical_g_obs(
    g_obs: Any,
    config_or_data_cfg: Mapping[str, Any],
    *,
    resolution: int,
    context: str = "g_obs",
) -> None:
    if not isinstance(g_obs, torch.Tensor):
        raise TypeError(f"{context} must be a torch.Tensor, got {type(g_obs).__name__}")
    want = expected_g_obs_shape(config_or_data_cfg, resolution)
    if tuple(g_obs.shape) != want:
        raise ValueError(f"{context} shape mismatch: expected {want}, got {tuple(g_obs.shape)}")


def cache_ldos_schema_metadata(config_or_data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = config_or_data_cfg
    data_cfg = _as_data_cfg(config_or_data_cfg)
    structural_cfg = {}
    potential_cfg = {}
    if isinstance(cfg, Mapping):
        potential_cfg = cfg.get("potential_sampler", {}) if isinstance(cfg.get("potential_sampler"), Mapping) else {}
        if isinstance(potential_cfg, Mapping):
            structural_cfg = potential_cfg.get("structural", {}) if isinstance(potential_cfg.get("structural"), Mapping) else {}
    target_rep = str(data_cfg.get("target_representation", "ldos_ab"))
    contains_defect_meta = bool(structural_cfg.get("enabled", False))
    contains_physics_meta = bool(
        data_cfg.get("return_physics_meta", False)
        or data_cfg.get("cache_require_physics_meta", False)
        or contains_defect_meta
    )
    base_schema = 2 if is_sublattice_resolved(data_cfg) else 1
    if contains_physics_meta or contains_defect_meta or target_rep != "ldos_ab":
        base_schema = max(base_schema, 3)
    meta: Dict[str, Any] = {
        "ldos_schema_version": base_schema,
        "target_representation": target_rep,
        "contains_physics_meta": contains_physics_meta,
        "contains_defect_meta": contains_defect_meta,
    }
    if is_sublattice_resolved(data_cfg):
        _require_layout_flags(data_cfg)
        meta.update(
            {
                "sublattice_resolved_ldos": True,
                "ldos_canonical_layout": _CANONICAL_LAYOUT,
                "ldos_model_layout": _MODEL_LAYOUT,
                "sublattice_count": 2,
            }
        )
    return meta


def require_graphene_if_sublattice_resolved(config: Mapping[str, Any]) -> None:
    if not is_sublattice_resolved(config):
        return
    ham_cfg = config.get("physics", {}).get("hamiltonian", {}) if isinstance(config, Mapping) else {}
    lattice_type = str(ham_cfg.get("type", "square_lattice")).lower()
    if lattice_type not in ("graphene", "honeycomb", "random"):
        raise ValueError(
            "data.sublattice_resolved_ldos=true is only supported for graphene/honeycomb in Phase 1."
        )
