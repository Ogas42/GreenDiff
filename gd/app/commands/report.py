from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


def _iter_json_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".json"):
                yield os.path.join(dirpath, name)


def _load_records(runs_root: str, task_filter: str | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _iter_json_files(runs_root):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict) or "metrics" not in data or "task" not in data:
            continue
        if task_filter and data.get("task") != task_filter:
            continue
        data = dict(data)
        data["_source_json"] = path
        records.append(data)
    return records


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "task": rec.get("task"),
        "dataset_suite": rec.get("dataset_suite"),
        "split": rec.get("split"),
        "seed": rec.get("seed"),
        "run_id": rec.get("run_id"),
        "variant": rec.get("variant") or rec.get("meta", {}).get("variant") or "baseline",
        "config_hash": rec.get("config_hash"),
        "checkpoint_tag": rec.get("checkpoint_tag"),
        "timestamp": rec.get("timestamp"),
        "source_json": rec.get("_source_json"),
    }
    for k, v in (rec.get("metrics") or {}).items():
        flat[f"metric__{k}"] = v
    return flat


def _write_csv(rows: List[Dict[str, Any]], out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def aggregate_results(runs_root: str, out: str, task_filter: str = "green_eval", group_by: str = "variant") -> str:
    records = _load_records(runs_root, task_filter=task_filter)
    flat = [_flatten_record(r) for r in records]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in flat:
        groups[str(row.get(group_by, "unknown"))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for g, rows in sorted(groups.items()):
        summary: Dict[str, Any] = {group_by: g, "n": len(rows)}
        numeric_cols = set()
        for row in rows:
            for k, v in row.items():
                if k.startswith("metric__") and isinstance(v, (int, float)) and not isinstance(v, bool):
                    if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        numeric_cols.add(k)
        for col in sorted(numeric_cols):
            vals = [float(r[col]) for r in rows if isinstance(r.get(col), (int, float))]
            if not vals:
                continue
            summary[f"{col}__mean"] = mean(vals)
            summary[f"{col}__std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out_rows.append(summary)
    return _write_csv(out_rows, out)


def plot_results(results_csv: str, out_dir: str, metric: str = "metric__rel_l2__mean") -> str:
    import matplotlib.pyplot as plt

    rows: List[Dict[str, str]] = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("No rows in aggregated results CSV.")
    if metric not in rows[0]:
        candidates = [k for k in rows[0].keys() if k.startswith("metric__")]
        raise KeyError(f"Metric '{metric}' not found. Available: {candidates}")

    labels = [r.get("variant", "unknown") for r in rows]
    values = [float(r[metric]) for r in rows]
    std_key = metric.replace("__mean", "__std") if metric.endswith("__mean") else None
    errs = [float(r.get(std_key, 0.0)) for r in rows] if std_key and std_key in rows[0] else None

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{metric.replace('__', '_')}.png")
    plt.figure(figsize=(10, 4.5))
    x = list(range(len(labels)))
    plt.bar(x, values, yerr=errs, capsize=4)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title("Aggregated Benchmark Results")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return fig_path


def add_subparser(subparsers):
    parser = subparsers.add_parser("report", help="Aggregate and plot benchmark/eval results")
    report_sub = parser.add_subparsers(dest="report_cmd", required=True)

    agg = report_sub.add_parser("aggregate", help="Aggregate result JSON files under runs root")
    agg.add_argument("--runs-root", required=True)
    agg.add_argument("--out", required=True)
    agg.add_argument("--task", default="green_eval")
    agg.add_argument("--group-by", default="variant")
    agg.set_defaults(handler=handle_report_aggregate)

    plots = report_sub.add_parser("plots", help="Create plots from aggregate CSV")
    plots.add_argument("--results", required=True)
    plots.add_argument("--out-dir", required=True)
    plots.add_argument("--metric", default="metric__rel_l2__mean")
    plots.set_defaults(handler=handle_report_plots)
    return parser


def handle_report_aggregate(args):
    out = aggregate_results(args.runs_root, args.out, task_filter=args.task, group_by=args.group_by)
    print(f"Aggregated results saved to {out}")
    return 0


def handle_report_plots(args):
    metric = args.metric
    try:
        fig = plot_results(args.results, args.out_dir, metric=metric)
    except KeyError:
        if metric == "metric__rel_l2__mean":
            fallback = "metric__mse__mean"
            fig = plot_results(args.results, args.out_dir, metric=fallback)
            print(f"Requested metric {metric!r} not found; fell back to {fallback!r}")
        else:
            raise
    print(f"Plot saved to {fig}")
    return 0
