from __future__ import annotations

from gd.core.config.loader import load_config
from gd.core.config.overrides import apply_profile
from gd.core.runtime.context import build_runtime_context


def add_subparser(subparsers):
    parser = subparsers.add_parser("eval", help="Run evaluation/visualization targets")
    parser.add_argument("target", choices=["green", "diffusion", "teacher_vis", "latent_green"])
    parser.add_argument("--config", default="gd/configs/default.yaml")
    parser.add_argument("--ckpt-dir", dest="ckpt_dir", default=None)
    parser.add_argument("--save-json", dest="save_json", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vis-n", type=int, default=None)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--profile", default="none", choices=["none", "local_4060", "local_4060_smoke", "remote_a6000"])
    parser.set_defaults(handler=handle_eval)
    return parser


def handle_eval(args):
    cfg = apply_profile(load_config(args.config), args.profile)
    ctx = build_runtime_context(cfg, init_process_group=False)
    mod_name = {
        "green": "gd.eval.green_eval",
        "latent_green": "gd.eval.green_eval",
        "diffusion": "gd.eval.diffusion_eval",
        "teacher_vis": "gd.eval.teacher_vis",
    }[args.target]
    module = __import__(mod_name, fromlist=["run"])
    result = module.run(
        config=cfg,
        runtime_ctx=ctx,
        ckpt_dir=args.ckpt_dir,
        save_json=args.save_json,
        output_dir=args.output_dir,
        split=args.split,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        vis_n=args.vis_n,
        use_ema=args.use_ema,
    )
    if isinstance(result, dict) and "metrics" in result:
        print(f"Eval {args.target} finished. metrics={result['metrics']}")
    else:
        print(f"Eval {args.target} finished.")
    return 0
