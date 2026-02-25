from __future__ import annotations

from gd.core.typing.types import ResumeState, StageResult
from gd.trainers.base import StageTrainer


class StudentTrainer(StageTrainer):
    stage_name = "student"
    requires = ["vae", "green", "diffusion"]

    def run(self, ctx, cfg, components, resume_state: ResumeState) -> StageResult:
        from gd.train.train_student import train_student

        train_student(cfg)
        return StageResult(stage=self.stage_name, success=True, message="Student training completed")

