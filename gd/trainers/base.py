from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from gd.core.typing.types import ResumeState, StageResult


class StageTrainer(ABC):
    stage_name: str = ""
    requires: list[str] = []

    def build(self, ctx: Any, cfg: Dict[str, Any]) -> Any:
        return None

    def resume(self, ctx: Any, cfg: Dict[str, Any], components: Any) -> ResumeState:
        return ResumeState(step=0)

    @abstractmethod
    def run(self, ctx: Any, cfg: Dict[str, Any], components: Any, resume_state: ResumeState) -> StageResult:
        raise NotImplementedError

    def evaluate_after_train(self, ctx: Any, cfg: Dict[str, Any], result: StageResult) -> None:
        return None

    def run_stage(self, ctx: Any, cfg: Dict[str, Any]) -> dict:
        components = self.build(ctx, cfg)
        resume_state = self.resume(ctx, cfg, components)
        result = self.run(ctx, cfg, components, resume_state)
        self.evaluate_after_train(ctx, cfg, result)
        return {
            "stage": result.stage,
            "success": result.success,
            "step": result.step,
            "metrics": result.metrics,
            "artifacts": result.artifacts,
            "message": result.message,
        }

