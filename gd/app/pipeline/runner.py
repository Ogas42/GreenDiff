from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import yaml

from gd.app.pipeline.hooks import run_post_stage_hooks
from gd.app.registry.stages import build_stage_registry, resolve_stage_order
from gd.core.checkpoints.manager import CheckpointManager
from gd.core.config.loader import load_config, resolve_config_paths
from gd.core.runtime.context import build_runtime_context


def _timestamp_dir(base_dir: str) -> str:
    return os.path.join(base_dir, dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


def _save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


@dataclass
class PipelineRunSummary:
    run_dir: str
    stages: List[str]
    results: Dict[str, Any]


class PipelineRunner:
    def __init__(self) -> None:
        self.registry = build_stage_registry()

    def prepare_config(self, config_path: str) -> Dict[str, Any]:
        return load_config(config_path)

    def prepare_run(self, config: Dict[str, Any], init_from: Optional[str] = None, run_name: Optional[str] = None):
        base_runs_dir = config["paths"]["runs_root"]
        run_dir = os.path.join(base_runs_dir, run_name) if run_name else _timestamp_dir(base_runs_dir)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        cfg = dict(config)
        cfg["paths"] = dict(config["paths"])
        cfg["paths"]["workdir"] = run_dir
        cfg["paths"]["checkpoints"] = ckpt_dir
        cfg["paths"]["logs"] = log_dir
        cfg = resolve_config_paths(cfg, cfg["paths"].get("config_path"))
        _save_config(cfg, os.path.join(run_dir, "config.yaml"))

        ctx = build_runtime_context(cfg, init_process_group=True)
        ckpt_mgr = CheckpointManager(cfg["paths"]["runs_root"], cfg["paths"]["checkpoints"])
        if init_from:
            ckpt_mgr.copy_from_run(init_from, ["*.pt"])
        return cfg, ctx, ckpt_mgr

    def run_with_config(
        self,
        config: Dict[str, Any],
        stages: Iterable[str],
        init_from: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> PipelineRunSummary:
        ordered = resolve_stage_order(stages, self.registry)
        cfg, ctx, _ckpt_mgr = self.prepare_run(config, init_from=init_from, run_name=run_name)
        results: Dict[str, Any] = {}
        for stage in ordered:
            handler = self.registry[stage]
            results[stage] = handler.runner(ctx, cfg)
            run_post_stage_hooks(stage, ctx, cfg, results[stage])
        return PipelineRunSummary(run_dir=cfg["paths"]["workdir"], stages=ordered, results=results)

    def run(self, config_path: str, stages: Iterable[str], init_from: Optional[str] = None) -> PipelineRunSummary:
        cfg = self.prepare_config(config_path)
        return self.run_with_config(cfg, stages=stages, init_from=init_from)
