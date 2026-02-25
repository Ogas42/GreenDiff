from __future__ import annotations

from gd.app.pipeline.runner import PipelineRunner
from gd.core.config.overrides import apply_profile


def add_subparser(subparsers):
    parser = subparsers.add_parser("pipeline", help="Pipeline operations")
    pipeline_sub = parser.add_subparsers(dest="pipeline_cmd", required=True)
    run_parser = pipeline_sub.add_parser("run", help="Run pipeline stages")
    run_parser.add_argument("--config", default="gd/configs/default.yaml")
    run_parser.add_argument("--stages", nargs="+", required=True, choices=["data", "vae", "green", "diffusion", "student"])
    run_parser.add_argument("--init-from", dest="init_from", default=None)
    run_parser.add_argument("--profile", default="none", choices=["none", "local_4060", "local_4060_smoke", "remote_a6000"])
    run_parser.add_argument("--run-name", default=None)
    run_parser.set_defaults(handler=handle_pipeline_run)
    return parser


def handle_pipeline_run(args):
    runner = PipelineRunner()
    cfg = apply_profile(runner.prepare_config(args.config), args.profile)
    summary = runner.run_with_config(cfg, stages=args.stages, init_from=args.init_from, run_name=args.run_name)
    print(f"Pipeline finished. run_dir={summary.run_dir}; stages={','.join(summary.stages)}")
    return 0
