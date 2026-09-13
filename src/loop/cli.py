"""`python -m loop` CLI: run, plan-once, execute-once, show-whitelist, submit-if-improved.

``run`` rewinds to the CSV-best CV, then plan → train → RESULTS+CSV → commit →
submit-if-best. ``execute-once`` records, commits, and may submit the current spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loop.config import find_root, load_settings
from loop.log import log
from loop.orchestrator import Loop


def main(argv: list[str] | None = None) -> int:
    """Parse argv, load Settings, and dispatch to a Loop method."""

    args = _parser().parse_args(argv)
    if args.cmd == "submit-if-improved":
        from loop.submit_if_improved import run_from_args

        return run_from_args(args)
    root = find_root(getattr(args, "root", None))
    config_path = Path(args.config).expanduser() if getattr(args, "config", None) else None
    if config_path and not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    settings = load_settings(root, config_path)
    loop = Loop(settings, dry_run=bool(getattr(args, "dry_run", False)))

    if args.cmd == "run":
        n = args.iterations if args.iterations is not None else settings.max_iterations
        return loop.run(n)
    if args.cmd == "plan-once":
        plan = loop.plan_once()
        log(f"planned {plan.strategy_id}: {plan.one_liner}")
        return 0
    if args.cmd == "execute-once":
        result = loop.execute_once()
        loop.record(result)
        loop.finish_run(result)
        log(f"executed {result.strategy_id}: {result.status}")
        return 0 if result.status == "ok" else 1
    if args.cmd == "show-whitelist":
        view = loop.show_whitelist()
        print(f"root: {settings.root}")
        print(f"bytes: {view.total_bytes}")
        print("included:")
        for item in view.listing:
            print(f"  {item}")
        if view.skipped:
            print("skipped:")
            for item in view.skipped:
                print(f"  {item}")
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Shared --root / --config / --dry-run flags for every subcommand."""

    parser.add_argument("--root", help="Loop repo root (ledgers, prompts, config).")
    parser.add_argument("--config", help="Path to loop.yaml (default: <root>/config/loop.yaml).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock planner/executor (no agy, qwen, GPU, or network).",
    )


def _parser() -> argparse.ArgumentParser:
    """Build the argparse tree for `python -m loop`."""

    p = argparse.ArgumentParser(
        prog="python -m loop",
        description="Token-thrifty Kaggle plan → execute → record loop.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="plan → execute → record for N iterations")
    _add_common(run)
    run.add_argument("--iterations", type=int, default=None, metavar="N")
    plan = sub.add_parser("plan-once", help="rewrite CURRENT_STRATEGY.md; append STRATEGIES.md")
    _add_common(plan)
    exe = sub.add_parser("execute-once", help="execute CURRENT_STRATEGY.md and record")
    _add_common(exe)
    show = sub.add_parser("show-whitelist", help="print files the planner would see")
    _add_common(show)
    sif = sub.add_parser(
        "submit-if-improved",
        help="submit + commit when CV (or LB) beats the CSV-best score",
    )
    from loop.submit_if_improved import add_arguments

    add_arguments(sif)
    return p


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
