from __future__ import annotations

import os
from typing import Any, Optional

from gd.core.typing.types import DistributedContext


def _safe_import_torch() -> Optional[Any]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def setup_distributed(config: dict, init_process_group: bool = False) -> DistributedContext:
    torch = _safe_import_torch()
    device_name = str(config.get("project", {}).get("device", "cpu"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    backend = None
    resolved_device = device_name

    if torch is not None:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        resolved_device = str(device)
        if init_process_group and is_distributed:
            backend = "nccl" if device.type == "cuda" else "gloo"
            try:
                import torch.distributed as dist  # type: ignore

                if not dist.is_initialized():
                    dist.init_process_group(backend=backend)
            except Exception:
                pass

    return DistributedContext(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        is_distributed=is_distributed,
        is_main=(not is_distributed) or rank == 0,
        backend=backend,
        device=resolved_device,
    )


def destroy_distributed() -> None:
    torch = _safe_import_torch()
    if torch is None:
        return
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        return

