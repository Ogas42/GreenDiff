from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


STAGE_NAMES = ("data", "vae", "green", "diffusion", "student")


@dataclass
class StageHandler:
    name: str
    requires: List[str]
    runner: Callable[..., object]
    kind: str = "trainer"


def _data_stage_runner(ctx, cfg):
    from gd.data.dataset import generate_cache

    generate_cache(cfg, ["train", "val"])
    return {"stage": "data", "success": True}


def build_stage_registry() -> Dict[str, StageHandler]:
    from gd.trainers.diffusion_trainer import DiffusionTrainer
    from gd.trainers.latent_green_trainer import LatentGreenTrainer
    from gd.trainers.student_trainer import StudentTrainer
    from gd.trainers.vae_trainer import VAETrainer

    return {
        "data": StageHandler(name="data", requires=[], runner=_data_stage_runner, kind="data"),
        "vae": StageHandler(name="vae", requires=[], runner=VAETrainer().run_stage),
        "green": StageHandler(name="green", requires=["vae"], runner=LatentGreenTrainer().run_stage),
        "diffusion": StageHandler(name="diffusion", requires=["vae", "green"], runner=DiffusionTrainer().run_stage),
        "student": StageHandler(name="student", requires=["vae", "green", "diffusion"], runner=StudentTrainer().run_stage),
    }


def resolve_stage_order(selected: Iterable[str], registry: Dict[str, StageHandler] | None = None) -> List[str]:
    registry = registry or build_stage_registry()
    selected_list = list(selected)
    seen = set()
    out: List[str] = []

    def visit(stage: str) -> None:
        if stage in seen:
            return
        if stage not in registry:
            raise KeyError(f"Unknown stage: {stage}")
        for dep in registry[stage].requires:
            if dep in selected_list:
                visit(dep)
        seen.add(stage)
        out.append(stage)

    for stage in selected_list:
        visit(stage)
    return out

