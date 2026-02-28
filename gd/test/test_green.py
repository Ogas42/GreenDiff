import argparse
import os
import sys
from typing import Any, Dict, Optional


def _ensure_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()

from gd.eval import green_eval  # noqa: E402
from gd.utils.config_utils import load_config, resolve_config_paths  # noqa: E402
from gd.utils.ldos_transform import force_linear_ldos_mode  # noqa: E402


def _fmt(x: Optional[float], spec: str) -> str:
    if x is None:
        return "na"
    return format(float(x), spec)


def _metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    v = metrics.get(key)
    return None if v is None else float(v)


def _print_report(result: Dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    print("\n" + "=" * 60)
    print("Linear Evaluation Results")
    print("=" * 60)
    print(f"{'MSE (Linear)':<30}: {_fmt(_metric(metrics, 'mse'), '.2e')}")
    print(f"{'MAE (Linear)':<30}: {_fmt(_metric(metrics, 'mae'), '.2e')}")
    print(f"{'Rel L2 (Linear)':<30}: {_fmt(_metric(metrics, 'rel_l2'), '.4f')}")
    print("-" * 60)
    print(f"{'Peak Ratio mean':<30}: {_fmt(_metric(metrics, 'peak_ratio_mean'), '.4f')}")
    print(f"{'Peak Ratio p95':<30}: {_fmt(_metric(metrics, 'peak_ratio_p95'), '.4f')}")
    print(f"{'Peak Ratio p99':<30}: {_fmt(_metric(metrics, 'peak_ratio_p99'), '.4f')}")
    print(f"{'Pred/Obs P99 ratio':<30}: {_fmt(_metric(metrics, 'pred_p99_over_obs_p99'), '.4f')}")
    print(f"{'Pred/Obs Std ratio':<30}: {_fmt(_metric(metrics, 'pred_std_over_obs_std'), '.4f')}")
    print(f"{'Pred/Obs Mean ratio':<30}: {_fmt(_metric(metrics, 'pred_obs_mean_ratio_mean'), '.4f')}")
    print(f"{'Pred/Obs Max ratio':<30}: {_fmt(_metric(metrics, 'pred_obs_max_ratio_mean'), '.4f')}")
    print("-" * 60)
    print(f"{'PSD error':<30}: {_fmt(_metric(metrics, 'psd_error'), '.4f')}")
    print(f"{'Residual':<30}: {_fmt(_metric(metrics, 'residual'), '.2e')}")
    print(f"{'Hopping mean':<30}: {_fmt(_metric(metrics, 'hopping_mean_mean'), '.4f')}")
    print(f"{'Hopping std':<30}: {_fmt(_metric(metrics, 'hopping_std_mean'), '.4f')}")
    print("=" * 60)


def _enforce_thresholds(
    result: Dict[str, Any],
    *,
    rel_l2_max: Optional[float],
    peak_ratio_max: Optional[float],
    peak_ratio_p95_max: Optional[float],
) -> None:
    metrics = result.get("metrics", {})
    errors = []
    if rel_l2_max is not None:
        rel = _metric(metrics, "rel_l2")
        if rel is None or rel > rel_l2_max:
            errors.append(f"rel_l2={rel} > {rel_l2_max}")
    if peak_ratio_max is not None:
        pr = _metric(metrics, "peak_ratio_mean")
        if pr is None or pr > peak_ratio_max:
            errors.append(f"peak_ratio_mean={pr} > {peak_ratio_max}")
    if peak_ratio_p95_max is not None:
        pr95 = _metric(metrics, "peak_ratio_p95")
        if pr95 is None or pr95 > peak_ratio_p95_max:
            errors.append(f"peak_ratio_p95={pr95} > {peak_ratio_p95_max}")
    if errors:
        raise SystemExit("Green eval threshold check failed: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Green operator evaluation (unified)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    parser.add_argument("--ckpt_dir", default=None, help="Checkpoint directory override")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_json", default=None, help="Where to save eval json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rel_l2_max", type=float, default=None, help="Optional fail threshold")
    parser.add_argument("--peak_ratio_max", type=float, default=None, help="Optional fail threshold (mean)")
    parser.add_argument("--peak_ratio_p95_max", type=float, default=None, help="Optional fail threshold (p95)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    force_linear_ldos_mode(cfg, verbose=True, context="test_green")
    cfg = resolve_config_paths(cfg, args.config)

    result = green_eval.run(
        cfg,
        ckpt_dir=args.ckpt_dir,
        split=args.split,
        max_batches=args.num_batches,
        batch_size=args.batch_size,
        save_json=args.save_json,
        quiet=args.quiet,
    )
    _print_report(result)
    _enforce_thresholds(
        result,
        rel_l2_max=args.rel_l2_max,
        peak_ratio_max=args.peak_ratio_max,
        peak_ratio_p95_max=args.peak_ratio_p95_max,
    )


if __name__ == "__main__":
    main()
