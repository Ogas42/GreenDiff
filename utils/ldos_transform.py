from typing import Any, Dict, Mapping, Optional

import torch


def _as_data_cfg(config_or_data_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if "data" in config_or_data_cfg and isinstance(config_or_data_cfg.get("data"), Mapping):
        return config_or_data_cfg["data"]
    return config_or_data_cfg


def get_ldos_transform_cfg(config_or_data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = _as_data_cfg(config_or_data_cfg)
    ldos_cfg = data_cfg.get("ldos_transform", {})
    return ldos_cfg if isinstance(ldos_cfg, dict) else {}


def ldos_transform_enabled(config_or_data_cfg: Mapping[str, Any]) -> bool:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    return bool(ldos_cfg.get("enabled", False))


def ldos_log_enabled(config_or_data_cfg: Mapping[str, Any]) -> bool:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    log_cfg = ldos_cfg.get("log", {})
    return bool(ldos_cfg.get("enabled", False) and isinstance(log_cfg, dict) and log_cfg.get("enabled", False))


def ldos_quantile_enabled(config_or_data_cfg: Mapping[str, Any]) -> bool:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    quant_cfg = ldos_cfg.get("quantile", {})
    return bool(ldos_cfg.get("enabled", False) and isinstance(quant_cfg, dict) and quant_cfg.get("enabled", False))


def ldos_log_eps(config_or_data_cfg: Mapping[str, Any]) -> float:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    log_cfg = ldos_cfg.get("log", {})
    if not isinstance(log_cfg, dict):
        return 1.0e-6
    return float(log_cfg.get("eps", 1.0e-6))


def ldos_scale(config_or_data_cfg: Mapping[str, Any]) -> float:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    return float(ldos_cfg.get("scale", 1.0))


def ldos_transform_signature(config_or_data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    ldos_cfg = get_ldos_transform_cfg(config_or_data_cfg)
    return {
        "enabled": bool(ldos_cfg.get("enabled", False)),
        "apply_to_cache": bool(ldos_cfg.get("apply_to_cache", True)),
        "log_enabled": ldos_log_enabled(config_or_data_cfg),
        "log_eps": round(ldos_log_eps(config_or_data_cfg), 12),
        "quantile_enabled": ldos_quantile_enabled(config_or_data_cfg),
        "scale": round(ldos_scale(config_or_data_cfg), 12),
    }


def _ensure_no_quantile(config_or_data_cfg: Mapping[str, Any]) -> None:
    if ldos_quantile_enabled(config_or_data_cfg):
        raise RuntimeError(
            "Quantile LDOS transform is not supported in the unified model/inference transform path. "
            "Use linear-only LDOS mode (log=false, quantile=false)."
        )


def ldos_obs_from_linear(g_lin: torch.Tensor, config_or_data_cfg: Mapping[str, Any]) -> torch.Tensor:
    """Map physical linear LDOS to the observation domain used by the dataset."""
    x = g_lin
    if not ldos_transform_enabled(config_or_data_cfg):
        return x
    _ensure_no_quantile(config_or_data_cfg)
    if ldos_log_enabled(config_or_data_cfg):
        eps = ldos_log_eps(config_or_data_cfg)
        x = torch.log(x.clamp_min(0) + eps)
    scale = ldos_scale(config_or_data_cfg)
    if scale != 1.0:
        x = x * scale
    return x


def ldos_linear_from_obs(g_obs: torch.Tensor, config_or_data_cfg: Mapping[str, Any]) -> torch.Tensor:
    """Invert dataset observation-domain LDOS back to physical linear LDOS."""
    x = g_obs
    if not ldos_transform_enabled(config_or_data_cfg):
        return x
    _ensure_no_quantile(config_or_data_cfg)
    scale = ldos_scale(config_or_data_cfg)
    if scale != 1.0:
        x = x / scale
    if ldos_log_enabled(config_or_data_cfg):
        eps = ldos_log_eps(config_or_data_cfg)
        x = (x.exp() - eps).clamp_min(0)
    return x


def force_linear_ldos_mode(
    config: Dict[str, Any],
    *,
    default_scale: Optional[float] = 1.0e5,
    ensure_cache_scaled_flag: bool = True,
    verbose: bool = False,
    context: str = "",
) -> Dict[str, Any]:
    """
    Mutates config to a linear-only LDOS observation pipeline.

    - Disables log / quantile transforms.
    - Keeps linear `scale` (or assigns a default when missing).
    - Marks `cache_scaled=true` by default so cached samples are treated as already transformed.
    """
    data_cfg = config.setdefault("data", {})
    ldos_cfg = data_cfg.setdefault("ldos_transform", {})
    changes = {}

    if "enabled" not in ldos_cfg:
        ldos_cfg["enabled"] = False

    log_cfg = ldos_cfg.setdefault("log", {})
    if log_cfg.get("enabled", False):
        log_cfg["enabled"] = False
        changes["log.enabled"] = False
    log_cfg.setdefault("eps", 1.0e-6)

    quant_cfg = ldos_cfg.setdefault("quantile", {})
    if quant_cfg.get("enabled", False):
        quant_cfg["enabled"] = False
        changes["quantile.enabled"] = False
    quant_cfg.setdefault("eps", 1.0e-6)

    if "scale" not in ldos_cfg and default_scale is not None:
        ldos_cfg["scale"] = float(default_scale)
        changes["scale"] = float(default_scale)
    if ensure_cache_scaled_flag and "cache_scaled" not in ldos_cfg and bool(ldos_cfg.get("apply_to_cache", True)):
        ldos_cfg["cache_scaled"] = True
        changes["cache_scaled"] = True

    if verbose and changes:
        prefix = f"[{context}] " if context else ""
        print(f"{prefix}Forced linear LDOS mode: {changes}")
    return changes

