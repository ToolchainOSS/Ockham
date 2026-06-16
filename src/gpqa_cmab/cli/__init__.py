"""``gpqa-cmab`` command-line entrypoint.

The CLI is split across this package: ``parser`` builds the argument parser,
``commands_*`` implement the subcommand handlers, and ``support`` holds the
shared client/cost/manifest plumbing. ``main`` dispatches and installs the
run-scoped telemetry recorder.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from gpqa_cmab.cli.parser import build_parser
from gpqa_cmab.cli.support import _CLI_TO_ENV, _apply_cli_overrides, make_client
from gpqa_cmab.config import clear_settings_cache, load_dotenv
from gpqa_cmab.telemetry_db import TelemetryRecorder, open_backend, set_recorder

if TYPE_CHECKING:
    import argparse

__all__ = [
    "_CLI_TO_ENV",
    "_apply_cli_overrides",
    "build_parser",
    "main",
    "make_client",
]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    args._gpqa_argv = [parser.prog, *raw_argv]  # noqa: SLF001 — internal argv passthrough
    if getattr(args, "env_file", None) is not None:
        loaded = load_dotenv(args.env_file, override=True)
        if loaded is None:
            raise SystemExit(f"--env-file not found: {args.env_file}")
        clear_settings_cache()
    # Install the durable telemetry recorder for the duration of this CLI
    # invocation. ``telemetry-db`` subcommands are read-only and don't need
    # a run-scoped recorder.
    command = _command_name(args)
    if _is_telemetry_db_command(args, command):
        args.func(args)
        return
    recorder = TelemetryRecorder(open_backend())
    set_recorder(recorder)
    try:
        with recorder.run(command=command, argv=raw_argv):
            args.func(args)
    finally:
        set_recorder(None)


def _command_name(args: argparse.Namespace) -> str:
    func = getattr(args, "func", None)
    if func is None:
        return "unknown"
    name = getattr(func, "__name__", "unknown")
    if name.startswith("cmd_"):
        return name[len("cmd_") :].replace("_", "-")
    return name


def _is_telemetry_db_command(args: argparse.Namespace, command: str) -> bool:
    if command == "telemetry-db" or hasattr(args, "telemetry_db_action"):
        return True
    func = getattr(args, "func", None)
    return func is not None and getattr(func, "__module__", "").endswith(
        "telemetry_db_cli"
    )


if __name__ == "__main__":
    main()
