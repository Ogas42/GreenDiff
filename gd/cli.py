from __future__ import annotations

import argparse
import sys

from gd.app.commands import benchmark as benchmark_cmd
from gd.app.commands import eval as eval_cmd
from gd.app.commands import pipeline as pipeline_cmd
from gd.app.commands import report as report_cmd
from gd.app.commands import train as train_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gd unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pipeline_cmd.add_subparser(subparsers)
    train_cmd.add_subparser(subparsers)
    eval_cmd.add_subparser(subparsers)
    benchmark_cmd.add_subparser(subparsers)
    report_cmd.add_subparser(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
