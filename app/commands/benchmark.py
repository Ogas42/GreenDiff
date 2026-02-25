from __future__ import annotations

import copy
import os
from typing import Any, Dict

from gd.app.pipeline.runner import PipelineRunner
from gd.core.config.overrides import apply_profile
from gd.core.logging.results import append_run_record_jsonl, load_json, save_eval_result_json


def _deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        _deep_set(out, k, v)
    return out


def _synthetic_main_v1_variants() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {},
        "no_fft_psd": {
            "latent_green.model.use_fft_loss": False,
            "latent_green.model.fft_loss_weight": 0.0,
            "latent_green.model.psd_loss_weight": 0.0,
        },
        "no_multiscale": {"latent_green.model.multiscale_loss_weight": 0.0},
        "small_model": {
            "latent_green.model.base_channels": 64,
            "latent_green.model.num_res_blocks": 2,
        },
        "no_residual": {"latent_green.model.residual_loss_weight": 0.0},
    }


def _synthetic_main_v1_diffusion_variants() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {},
        "no_phys": {
            "diffusion.training.phys_loss_weight": 0.0,
            "diffusion.training.consistency_loss_weight": 0.0,
        },
        "no_x0": {"diffusion.training.x0_loss_weight": 0.0},
        "no_minsnr": {"diffusion.training.min_snr.enabled": False},
        "no_ema": {"diffusion.training.ema.enabled": False},
        "no_psd_phys": {"diffusion.training.psd_loss_weight": 0.0},
    }


def add_subparser(subparsers):
    parser = subparsers.add_parser("benchmark", help="Run benchmark suites")
    bench_sub = parser.add_subparsers(dest="benchmark_cmd", required=True)
    run = bench_sub.add_parser("run", help="Run a benchmark suite")
    run.add_argument("--suite", default="synthetic_main_v1", choices=["synthetic_main_v1"])
    run.add_argument("--task", default="green", choices=["green", "diffusion"])
    run.add_argument("--config", default="gd/configs/default.yaml")
    run.add_argument("--profile", default="none", choices=["none", "local_4060", "local_4060_smoke", "remote_a6000"])
    run.add_argument("--mode", default="fast", choices=["fast", "full"], help="fast uses shorter eval / fewer variants")
    run.add_argument("--seeds", type=int, default=None, help="Number of seeds (0..N-1 offsets from base seed)")
    run.add_argument("--variants", nargs="*", default=None, help="Subset of variants")
    run.add_argument("--stages", nargs="*", default=None, choices=["data", "vae", "green", "diffusion", "student"])
    run.add_argument("--eval-split", default="val")
    run.add_argument("--eval-max-batches", type=int, default=None)
    run.add_argument("--eval-vis-n", type=int, default=None)
    run.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    run.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-train", action="store_true")
    run.set_defaults(handler=handle_benchmark_run)
    return parser


def handle_benchmark_run(args):
    from gd.core.config.loader import load_config
    from gd.eval.diffusion_eval import run as run_diffusion_eval
    from gd.eval.green_eval import run as run_green_eval

    base_cfg = apply_profile(load_config(args.config), args.profile)
    base_seed = int(base_cfg.get("project", {}).get("seed", 42))
    variants_all = _synthetic_main_v1_variants() if args.task == "green" else _synthetic_main_v1_diffusion_variants()
    default_variants = ["baseline", "no_residual"] if args.mode == "fast" else list(variants_all.keys())
    if args.task == "diffusion":
        default_variants = ["baseline", "no_phys"] if args.mode == "fast" else list(variants_all.keys())
    variants = args.variants or default_variants
    eval_max_batches = int(args.eval_max_batches if args.eval_max_batches is not None else (10 if args.mode == "fast" else 50))
    seed_count = int(args.seeds if args.seeds is not None else (1 if args.mode == "fast" else 3))
    stages = args.stages
    if not stages:
        stages = ["vae", "green"] if args.task == "green" else ["vae", "green", "diffusion"]
    runner = PipelineRunner()
    all_records_jsonl = os.path.join(base_cfg["paths"]["runs_root"], "benchmark_records.jsonl")
    eval_task_name = "green_eval" if args.task == "green" else "diffusion_eval"

    for variant in variants:
        if variant not in variants_all:
            raise KeyError(f"Unknown variant '{variant}'. Available: {list(variants_all)}")
        for seed_offset in range(seed_count):
            seed = base_seed + seed_offset
            cfg = _apply_overrides(base_cfg, variants_all[variant])
            cfg["project"]["seed"] = seed
            run_name = f"bench_{args.suite}_{variant}_seed{seed}"

            if args.dry_run:
                print(f"[DRY-RUN] task={args.task} suite={args.suite} variant={variant} seed={seed} stages={stages} run={run_name}")
                continue

            if not args.skip_train:
                summary = runner.run_with_config(cfg, stages=stages, run_name=run_name)
                run_dir = summary.run_dir
            else:
                run_dir = os.path.join(cfg["paths"]["runs_root"], run_name)

            ckpt_dir = os.path.join(run_dir, "checkpoints")
            metrics_dir = os.path.join(run_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            result_json = os.path.join(metrics_dir, f"{eval_task_name}_{args.eval_split}.json")
            if os.path.exists(result_json):
                eval_result = load_json(result_json)
            else:
                if args.task == "green":
                    eval_result = run_green_eval(
                        config=cfg,
                        ckpt_dir=ckpt_dir,
                        split=args.eval_split,
                        max_batches=eval_max_batches,
                        save_json=result_json,
                        output_dir=metrics_dir,
                        dataset_suite=args.suite,
                        variant=variant,
                        quiet=True,
                    )
                else:
                    eval_result = run_diffusion_eval(
                        config=cfg,
                        ckpt_dir=ckpt_dir,
                        split=args.eval_split,
                        max_batches=eval_max_batches,
                        save_json=result_json,
                        output_dir=metrics_dir,
                        dataset_suite=args.suite,
                        variant=variant,
                        quiet=True,
                        vis_n=args.eval_vis_n,
                        use_ema=bool(args.use_ema),
                    )
            eval_result["variant"] = variant
            save_eval_result_json(eval_result, result_json)
            append_run_record_jsonl(eval_result, all_records_jsonl)
            print(
                f"[benchmark] task={args.task} suite={args.suite} variant={variant} seed={seed} "
                f"run={run_name} rel_l2={eval_result.get('metrics', {}).get('rel_l2')}"
            )
    return 0
