from __future__ import annotations

from typing import Any, Optional

from gd.core.typing.types import AmpContext


def _safe_import_torch() -> Optional[Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def build_amp_context(config: dict, device: str) -> AmpContext:
    torch = _safe_import_torch()
    precision = str(config.get("project", {}).get("precision", "fp32"))
    if torch is None:
        return AmpContext(precision=precision)
    use_amp = str(device).startswith("cuda") and precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_scaler = use_amp and precision == "fp16"
    scaler = None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=use_scaler)
    return AmpContext(
        precision=precision,
        use_amp=use_amp,
        amp_dtype=amp_dtype if use_amp else None,
        use_scaler=use_scaler,
        scaler=scaler,
    )

