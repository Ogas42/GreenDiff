from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ProjectConfigView:
    raw: Dict[str, Any]

    @property
    def device(self) -> str:
        return str(self.raw.get("device", "cpu"))

    @property
    def precision(self) -> str:
        return str(self.raw.get("precision", "fp32"))


@dataclass(frozen=True)
class StageConfigView:
    raw: Dict[str, Any]

    @property
    def training(self) -> Dict[str, Any]:
        return dict(self.raw.get("training", {}))

