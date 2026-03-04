import argparse
import datetime as _dt
import os
import subprocess
import sys
from typing import Any, Dict, List

import yaml


def _ensure_project_root() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


PROJECT_ROOT = _ensure_project_root()

from gd.utils.config_utils import resolve_config_path  # noqa: E402


DEFAULT_BUNDLE = "gd/configs/suites/global_ablation_server.yaml"


def _load_yaml(path: str) -> Dict[str, Any]:
    resolved = resolve_config_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML: {resolved}")
    data["_resolved_path"] = resolved
    return data


def _suite_cfg(bundle: Dict[str, Any]) -> Dict[str, Any]:
    suite = bundle.get("suite")
    if not isinstance(suite, dict):
        raise ValueError("Missing 'suite' mapping in bundle config")
    return suite


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _build_batch_root(suite_cfg: Dict[str, Any], cli_out_dir: str | None) -> str:
    if cli_out_dir:
        return os.path.abspath(cli_out_dir)
    output_root = suite_cfg.get("output_root", "gd/runs/ablation_batches")
    output_root = resolve_config_path(str(output_root))
    batch_name = str(suite_cfg.get("name", "global_ablation_server"))
    batch_root = os.path.join(output_root, batch_name)
    if _bool(suite_cfg.get("include_timestamp", True), True):
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_root = os.path.join(batch_root, stamp)
    return os.path.abspath(batch_root)


def _run_subprocess(cmd: List[str], cwd: str, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    with open(log_path, "w", encoding="utf-8") as f:
        if completed.stdout:
            f.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                f.write("\n")
        if completed.stderr:
            f.write("\n[stderr]\n")
            f.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                f.write("\n")
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return int(completed.returncode)


def _write_batch_manifest(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_batch_readme(path: str, manifest: Dict[str, Any]) -> None:
    lines = [
        f"# {manifest['batch_name']}",
        "",
        f"- Bundle Config: `{manifest['bundle_config']}`",
        f"- Base Config: `{manifest['base_config']}`",
        f"- Stage: `{manifest['stage']}`",
        "",
        "## Suites",
        "",
        "| Suite | Status | Output Dir | Log |",
        "| --- | --- | --- | --- |",
    ]
    for item in manifest.get("suite_runs", []):
        lines.append(
            "| {suite} | {status} | `{out_dir}` | `{log_path}` |".format(
                suite=item["suite"],
                status=item["status"],
                out_dir=item["out_dir"],
                log_path=item["log_path"],
            )
        )
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="One-shot server wrapper for running multiple GreenDiff global ablation suites."
    )
    parser.add_argument(
        "--bundle",
        default=DEFAULT_BUNDLE,
        help="Bundle YAML describing the suite batch to run.",
    )
    parser.add_argument(
        "--stage",
        choices=("print", "train", "eval", "train_then_eval"),
        help="Override bundle stage.",
    )
    parser.add_argument("--out_dir", help="Override batch output directory.")
    parser.add_argument("--num_batches", type=int, help="Override eval.num_batches in bundle.")
    parser.add_argument("--batch_size", type=int, help="Override eval.batch_size in bundle.")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Force stop-on-error even if the bundle disables it.",
    )
    args = parser.parse_args()

    bundle = _load_yaml(args.bundle)
    suite_cfg = _suite_cfg(bundle)
    bundle_path = bundle["_resolved_path"]
    base_config = str(suite_cfg.get("base_config", "gd/configs/default.yaml"))
    suites = suite_cfg.get("suites", [])
    if not isinstance(suites, list) or not suites:
        raise ValueError("suite.suites must be a non-empty list")

    stage = args.stage or str(suite_cfg.get("stage", "train_then_eval"))
    num_batches = int(args.num_batches if args.num_batches is not None else suite_cfg.get("eval", {}).get("num_batches", 20))
    batch_size = int(args.batch_size if args.batch_size is not None else suite_cfg.get("eval", {}).get("batch_size", 4))
    stop_on_error = args.stop_on_error or _bool(suite_cfg.get("stop_on_error", True), True)
    verbose = _bool(suite_cfg.get("verbose", True), True)

    batch_root = _build_batch_root(suite_cfg, args.out_dir)
    os.makedirs(batch_root, exist_ok=True)
    logs_root = os.path.join(batch_root, "batch_logs")
    os.makedirs(logs_root, exist_ok=True)

    manifest: Dict[str, Any] = {
        "batch_name": str(suite_cfg.get("name", "global_ablation_server")),
        "description": str(suite_cfg.get("description", "")),
        "bundle_config": bundle_path,
        "base_config": os.path.abspath(resolve_config_path(base_config)),
        "stage": stage,
        "batch_root": batch_root,
        "suite_runs": [],
    }

    print("\n" + "=" * 100)
    print(f"One-Shot Global Ablation Batch: {manifest['batch_name']}")
    print(f"Bundle Config                : {bundle_path}")
    print(f"Base Config                  : {manifest['base_config']}")
    print(f"Stage                        : {stage}")
    print(f"Batch Root                   : {batch_root}")
    print("=" * 100)

    failures: List[str] = []
    for index, suite_name in enumerate(suites, start=1):
        suite_out_dir = os.path.join(batch_root, "suites", f"{index:02d}_{suite_name}")
        log_path = os.path.join(logs_root, f"{index:02d}_{suite_name}.log")
        cmd = [
            sys.executable,
            "-m",
            "gd.test.run_global_ablation",
            "--config",
            base_config,
            "--suite",
            str(suite_name),
            "--stage",
            stage,
            "--out_dir",
            suite_out_dir,
            "--num_batches",
            str(num_batches),
            "--batch_size",
            str(batch_size),
        ]
        if stop_on_error:
            cmd.append("--stop_on_error")

        if verbose:
            print(f"\n[{index:02d}] Suite: {suite_name}")
            print(f"Output Dir : {suite_out_dir}")
            print(f"Command    : {' '.join(cmd)}")

        rc = 0
        status = "PLANNED"
        if stage != "print":
            rc = _run_subprocess(cmd, PROJECT_ROOT, log_path)
            status = "OK" if rc == 0 else f"FAILED (exit={rc})"
        else:
            rc = _run_subprocess(cmd, PROJECT_ROOT, log_path)
            status = "PLANNED" if rc == 0 else f"FAILED (exit={rc})"

        manifest["suite_runs"].append(
            {
                "suite": str(suite_name),
                "out_dir": os.path.abspath(suite_out_dir),
                "log_path": os.path.abspath(log_path),
                "return_code": int(rc),
                "status": status,
            }
        )

        if rc != 0:
            failures.append(str(suite_name))
            if stop_on_error:
                break

    manifest_path = os.path.join(batch_root, "batch_manifest.yaml")
    readme_path = os.path.join(batch_root, "README.md")
    _write_batch_manifest(manifest_path, manifest)
    _write_batch_readme(readme_path, manifest)

    print("\n" + "=" * 100)
    print(f"Batch manifest written to: {manifest_path}")
    print(f"Batch README written to  : {readme_path}")
    print("=" * 100)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
