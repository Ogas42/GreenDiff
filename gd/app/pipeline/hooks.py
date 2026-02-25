from __future__ import annotations

import os
from typing import Any, Dict


def run_post_stage_hooks(stage: str, ctx: Any, config: Dict[str, Any], result: Any) -> None:
    if getattr(getattr(ctx, "dist", None), "is_main", True) is False:
        return
    if stage == "green":
        from gd.core.logging.results import append_run_record_jsonl
        from gd.eval.green_eval import run as run_green_eval

        metrics_dir = os.path.join(config["paths"]["workdir"], "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        save_json = os.path.join(metrics_dir, "green_eval_val.json")
        try:
            eval_result = run_green_eval(
                config=config,
                runtime_ctx=ctx,
                split="val",
                save_json=save_json,
                output_dir=metrics_dir,
            )
            append_run_record_jsonl(eval_result, os.path.join(metrics_dir, "eval_records.jsonl"))
        except Exception as e:
            print(f"[gd.pipeline.hooks] green post-eval failed: {e}")
    if stage == "diffusion":
        from gd.core.logging.results import append_run_record_jsonl
        from gd.eval.diffusion_eval import run as run_diffusion_eval

        metrics_dir = os.path.join(config["paths"]["workdir"], "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        save_json = os.path.join(metrics_dir, "diffusion_eval_val.json")
        try:
            eval_result = run_diffusion_eval(
                config=config,
                runtime_ctx=ctx,
                split="val",
                save_json=save_json,
                output_dir=metrics_dir,
                quiet=True,
            )
            append_run_record_jsonl(eval_result, os.path.join(metrics_dir, "eval_records.jsonl"))
        except Exception as e:
            print(f"[gd.pipeline.hooks] diffusion post-eval failed: {e}")
