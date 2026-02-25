from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def _json_default(obj: Any) -> str:
    return str(obj)


def _stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=_json_default, separators=(",", ":"))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_fingerprint(config: Dict[str, Any]) -> str:
    payload = _stable_json_dumps(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_fingerprint(ckpt_path: str | None) -> str | None:
    if not ckpt_path:
        return None
    try:
        st = os.stat(ckpt_path)
        payload = f"{os.path.basename(ckpt_path)}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        payload = os.path.basename(ckpt_path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_eval_result_json(result: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def append_run_record_jsonl(record: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(_stable_json_dumps(record))
        f.write("\n")
    return path


def append_train_metric_jsonl(record: Dict[str, Any], path: str) -> str:
    return append_run_record_jsonl(record, path)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            try:
                info["cuda_name_0"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        info["cuda_available"] = False
    return info
