from __future__ import annotations

import copy
import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def resolve_config_path(path: str) -> str:
    candidate = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(candidate):
        return os.path.abspath(candidate)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    project_root = _project_root()
    for trial in (os.path.join(project_root, candidate), os.path.join(project_root, "gd", candidate)):
        if os.path.exists(trial):
            return os.path.abspath(trial)
    if candidate.startswith("gd" + os.sep) or candidate.startswith("gd/"):
        trial = os.path.join(project_root, candidate)
        if os.path.exists(trial):
            return os.path.abspath(trial)
    return os.path.abspath(candidate)


def _resolve_path(value: Optional[str], base_dir: str) -> Optional[str]:
    if value is None:
        return None
    value = os.path.expandvars(os.path.expanduser(value))
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(base_dir, value))


def resolve_config_paths(config: Dict[str, Any], config_path: Optional[str] = None) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    project_root = _project_root()
    base_dir = project_root
    paths = config.setdefault("paths", {})
    paths["project_root"] = project_root
    if config_path:
        paths["config_path"] = os.path.abspath(config_path)

    workdir = paths.get("workdir") or os.path.join(project_root, "gd", "runs")
    paths["workdir"] = _resolve_path(workdir, base_dir)

    dataset_root = paths.get("dataset_root") or os.path.join(project_root, "data_cache")
    paths["dataset_root"] = _resolve_path(dataset_root, base_dir)

    runs_root = paths.get("runs_root")
    if runs_root is None:
        base_name = os.path.basename(os.path.normpath(paths["workdir"]))
        runs_root = os.path.dirname(paths["workdir"]) if base_name.startswith("20") else paths["workdir"]
    paths["runs_root"] = _resolve_path(runs_root, base_dir)

    checkpoints = paths.get("checkpoints")
    if checkpoints is None:
        checkpoints = os.path.join(paths["workdir"], "checkpoints")
    elif not os.path.isabs(os.path.expandvars(os.path.expanduser(checkpoints))):
        checkpoints = os.path.join(paths["workdir"], checkpoints)
    paths["checkpoints"] = _resolve_path(checkpoints, base_dir)

    logs = paths.get("logs")
    if logs is None:
        logs = os.path.join(paths["workdir"], "logs")
    elif not os.path.isabs(os.path.expandvars(os.path.expanduser(logs))):
        logs = os.path.join(paths["workdir"], logs)
    paths["logs"] = _resolve_path(logs, base_dir)
    return config


def load_config(path: str) -> Dict[str, Any]:
    resolved = resolve_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = resolve_config_paths(config, resolved)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    missing = [k for k in ("project", "paths", "data") if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    for key in ("workdir", "runs_root", "checkpoints", "logs"):
        if key not in config["paths"]:
            raise ValueError(f"Missing config.paths.{key}")


def get_stage_config(config: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    mapping = {
        "data": config.get("data", {}),
        "vae": config.get("vae", {}),
        "green": config.get("latent_green", {}),
        "latent_green": config.get("latent_green", {}),
        "diffusion": config.get("diffusion", {}),
        "student": config.get("student", {}),
    }
    if stage_name not in mapping:
        raise KeyError(f"Unknown stage: {stage_name}")
    return mapping[stage_name]


def get_latest_checkpoint_dir(work_dir: str, require_pattern: Optional[str] = None) -> Optional[str]:
    if not os.path.exists(work_dir):
        return None
    subdirs = [os.path.join(work_dir, d) for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    valid_runs = []
    for d in subdirs:
        ckpt_path = os.path.join(d, "checkpoints")
        if os.path.basename(d).startswith("20") and os.path.exists(ckpt_path):
            if require_pattern:
                if glob.glob(os.path.join(ckpt_path, require_pattern)):
                    valid_runs.append((d, ckpt_path))
            else:
                valid_runs.append((d, ckpt_path))
    if not valid_runs:
        return None
    valid_runs.sort(key=lambda x: os.path.basename(x[0]))
    return valid_runs[-1][1]


def build_paths_context(config: Dict[str, Any]):
    from gd.core.typing.types import PathsContext

    paths = config["paths"]
    return PathsContext(
        project_root=Path(paths["project_root"]),
        config_path=Path(paths["config_path"]) if paths.get("config_path") else None,
        runs_root=Path(paths["runs_root"]),
        workdir=Path(paths["workdir"]),
        checkpoints=Path(paths["checkpoints"]),
        logs=Path(paths["logs"]),
    )


