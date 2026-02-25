from __future__ import annotations

import os
import time
from typing import Any, Dict

import yaml

from gd.app.registry.stages import build_stage_registry
from gd.core.config.loader import load_config, resolve_config_paths
from gd.core.config.overrides import apply_profile
from gd.core.runtime.context import build_runtime_context


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", help="Run a single training stage")
    parser.add_argument("stage", choices=["vae", "green", "diffusion", "student"])
    parser.add_argument("--config", default="gd/configs/default.yaml")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", default="none", choices=["none", "local_4060", "local_4060_smoke", "remote_a6000"])
    parser.set_defaults(handler=handle_train)
    return parser


def _ensure_workdir(cfg: Dict[str, Any], workdir: str | None, run_name: str | None) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    if workdir is None and run_name:
        cfg["paths"]["workdir"] = os.path.join(cfg["paths"]["runs_root"], run_name)
        cfg["paths"]["checkpoints"] = os.path.join(cfg["paths"]["workdir"], "checkpoints")
        cfg["paths"]["logs"] = os.path.join(cfg["paths"]["workdir"], "logs")
    if workdir:
        cfg["paths"]["workdir"] = workdir
        cfg["paths"]["checkpoints"] = os.path.join(workdir, "checkpoints")
        cfg["paths"]["logs"] = os.path.join(workdir, "logs")
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs"], exist_ok=True)
    return resolve_config_paths(cfg, cfg["paths"].get("config_path"))


def handle_train(args):
    cfg = load_config(args.config)
    cfg = apply_profile(cfg, args.profile)
    if args.seed is not None:
        cfg["project"]["seed"] = int(args.seed)
    if args.run_name is None and args.workdir is None:
        args.run_name = f"{args.stage}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    cfg = _ensure_workdir(cfg, args.workdir, args.run_name)
    with open(os.path.join(cfg["paths"]["workdir"], "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    ctx = build_runtime_context(cfg, init_process_group=True)
    result = build_stage_registry()[args.stage].runner(ctx, cfg)
    print(f"Stage {args.stage} finished: {result}")
    return 0
