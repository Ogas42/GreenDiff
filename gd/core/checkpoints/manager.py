from __future__ import annotations

import glob
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

from gd.core.config.loader import get_latest_checkpoint_dir


def _ckpt_step_key(path: str) -> int:
    m = re.search(r"_step_(\d+)(?:_ema)?\.pt$", os.path.basename(path))
    if m:
        return int(m.group(1))
    try:
        return int(path.split("_")[-1].split(".")[0])
    except Exception:
        return -1


def normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        elif k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


@dataclass
class CheckpointManager:
    runs_root: str
    current_ckpt_dir: str

    def find_latest_in_current(self, pattern: str) -> Optional[str]:
        current = sorted(glob.glob(os.path.join(self.current_ckpt_dir, pattern)), key=_ckpt_step_key)
        return current[-1] if current else None

    def find_latest(self, pattern: str) -> Optional[str]:
        current = self.find_latest_in_current(pattern)
        if current:
            return current
        latest_dir = get_latest_checkpoint_dir(self.runs_root, require_pattern=pattern)
        if latest_dir is None:
            return None
        found = sorted(glob.glob(os.path.join(latest_dir, pattern)), key=_ckpt_step_key)
        return found[-1] if found else None

    def find_latest_with_fallback(self, pattern: str) -> Optional[str]:
        return self.find_latest(pattern)

    def copy_from_run(self, src_run_dir: str, patterns: list[str]) -> list[str]:
        src_ckpt = os.path.join(src_run_dir, "checkpoints")
        if not os.path.exists(src_ckpt):
            return []
        os.makedirs(self.current_ckpt_dir, exist_ok=True)
        copied: list[str] = []
        for pattern in patterns:
            for f in glob.glob(os.path.join(src_ckpt, pattern)):
                dst = os.path.join(self.current_ckpt_dir, os.path.basename(f))
                shutil.copy2(f, dst)
                copied.append(dst)
        return copied

    def save_state_dict(self, stage: str, step: int, state_dict: Dict[str, Any], torch_module: Any = None) -> str:
        os.makedirs(self.current_ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(self.current_ckpt_dir, f"{stage}_step_{step}.pt")
        if torch_module is None:
            import torch  # type: ignore

            torch_module = torch
        torch_module.save(state_dict, ckpt_path)
        return ckpt_path

    def load_state_dict(
        self,
        path: str,
        map_location: Any = None,
        normalize: bool = True,
        torch_module: Any = None,
    ) -> Dict[str, Any]:
        if torch_module is None:
            import torch  # type: ignore

            torch_module = torch
        try:
            state = torch_module.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch_module.load(path, map_location=map_location)
        if normalize and isinstance(state, dict):
            return normalize_state_dict_keys(state)
        return state
