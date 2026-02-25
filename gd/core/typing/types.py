from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PathsContext:
    project_root: Path
    config_path: Optional[Path]
    runs_root: Path
    workdir: Path
    checkpoints: Path
    logs: Path


@dataclass
class DistributedContext:
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    is_main: bool = True
    backend: Optional[str] = None
    device: str = "cpu"


@dataclass
class AmpContext:
    precision: str = "fp32"
    use_amp: bool = False
    amp_dtype: Any = None
    use_scaler: bool = False
    scaler: Any = None


@dataclass
class ResumeState:
    step: int = 0
    checkpoint_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    success: bool = True
    step: Optional[int] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class RuntimeContext:
    paths: PathsContext
    dist: DistributedContext
    amp: AmpContext
    raw_config: Dict[str, Any]

