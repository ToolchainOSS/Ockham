"""CLI subcommands for the durable telemetry DB.

Exposes ``gpqa-cmab telemetry-db {init,runs,tail,verify,reconstruct}``.
Lives in its own module so ``cli.py`` stays small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpqa_cmab.telemetry_db import (
    EventType,
    list_runs,
    open_backend,
    reconstruct_bandit_replay,
    reconstruct_factorial,
    reconstruct_run_summary,
    verify_integrity,
)


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "telemetry-db",
        help=(
            "Inspect and reconstruct research artefacts from the durable "
            "telemetry event store."
        ),
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "Override the database URL (default: $GPQA_TELEMETRY_DB_URL or "
            "sqlite:///artifacts/telemetry.sqlite)."
        ),
    )
    parser.set_defaults(func=_default)

    inner = parser.add_subparsers(dest="telemetry_db_action", required=False)

    init = inner.add_parser("init", help="Create tables/indexes if missing.")
    init.set_defaults(func=_cmd_init)

    runs = inner.add_parser("runs", help="List every recorded run.")
    runs.set_defaults(func=_cmd_runs)

    tail = inner.add_parser(
        "tail",
        help="Stream the latest N events (newest last) for quick inspection.",
    )
    tail.add_argument("--limit", type=int, default=20)
    tail.add_argument("--run-id", default=None)
    tail.add_argument(
        "--event-type",
        default=None,
        help="Filter by EventType value, e.g. 'llm.response'.",
    )
    tail.set_defaults(func=_cmd_tail)

    verify = inner.add_parser(
        "verify", help="Cheap integrity check (counts, duplicates, time bounds)."
    )
    verify.set_defaults(func=_cmd_verify)

    rebuild = inner.add_parser(
        "reconstruct",
        help="Rebuild research JSONL artefacts from the event store.",
    )
    rebuild.add_argument(
        "kind", choices=("factorial", "bandit", "summary"), default="summary"
    )
    rebuild.add_argument("--run-id", default=None)
    rebuild.add_argument("--output", type=Path, default=None)
    rebuild.set_defaults(func=_cmd_reconstruct)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _default(args: argparse.Namespace) -> None:
    # No subcommand given: behave like 'verify' so the tool is useful on its own.
    _cmd_verify(args)


def _cmd_init(args: argparse.Namespace) -> None:
    backend = open_backend(getattr(args, "db_url", None))
    backend.initialize()
    print(json.dumps({"status": "ok", "backend": backend.kind}))


def _cmd_runs(args: argparse.Namespace) -> None:
    backend = open_backend(getattr(args, "db_url", None))
    summaries = list_runs(backend)
    print(
        json.dumps(
            [s.model_dump(mode="json") for s in summaries],
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_tail(args: argparse.Namespace) -> None:
    backend = open_backend(getattr(args, "db_url", None))
    event_types = (
        [EventType(args.event_type)] if getattr(args, "event_type", None) else None
    )
    rows = list(
        backend.iter_events(
            run_id=args.run_id,
            event_types=event_types,
        )
    )
    for ev in rows[-args.limit :]:
        print(ev.model_dump_json())


def _cmd_verify(args: argparse.Namespace) -> None:
    backend = open_backend(getattr(args, "db_url", None))
    report = verify_integrity(backend)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    backend = open_backend(getattr(args, "db_url", None))
    out = args.output
    if args.kind == "factorial":
        out = out or Path("artifacts/reconstructed/full_factorial_results.jsonl")
        report = reconstruct_factorial(backend, out, run_id=args.run_id)
    elif args.kind == "bandit":
        if not args.run_id:
            raise SystemExit("--run-id is required for kind=bandit")
        out = out or Path(f"artifacts/reconstructed/bandit_{args.run_id}.jsonl")
        report = reconstruct_bandit_replay(backend, out, run_id=args.run_id)
    else:
        if not args.run_id:
            raise SystemExit("--run-id is required for kind=summary")
        report = reconstruct_run_summary(backend, args.run_id)
    print(report.model_dump_json(indent=2))


__all__ = ["register"]
