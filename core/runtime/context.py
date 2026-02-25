from __future__ import annotations

from gd.core.config.loader import build_paths_context
from gd.core.runtime.amp import build_amp_context
from gd.core.runtime.distributed import setup_distributed
from gd.core.typing.types import RuntimeContext


def build_runtime_context(config: dict, init_process_group: bool = False) -> RuntimeContext:
    dist_ctx = setup_distributed(config, init_process_group=init_process_group)
    amp_ctx = build_amp_context(config, device=dist_ctx.device)
    return RuntimeContext(
        paths=build_paths_context(config),
        dist=dist_ctx,
        amp=amp_ctx,
        raw_config=config,
    )

