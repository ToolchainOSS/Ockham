"""The ``train-gfn`` subcommand.

Kept isolated so importing the rest of the CLI never pulls in the optional
[gfn] extra (torch); the heavy imports happen lazily inside the handler.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gpqa_cmab.cli.support import (
    _manifest_argv,
    _manifest_path,
    _settings_manifest,
    _utc_now,
)
from gpqa_cmab.config import get_settings
from gpqa_cmab.telemetry import write_run_manifest

if TYPE_CHECKING:
    import argparse


def cmd_train_gfn(args: argparse.Namespace) -> None:
    started_utc = _utc_now()
    # Local import so the bare ``gpqa-cmab`` CLI does not require the heavy
    # [gfn] extra (torch) when the user only wants run-factorial / replay.
    try:
        from gpqa_cmab.gfn import CMABFilter, SubagentEnvironment
        from gpqa_cmab.gfn.training import train_cmab_gfn
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "train-gfn requires the [gfn] extra. Install with: uv sync --extra gfn"
        ) from exc

    env = SubagentEnvironment(temperature=args.temperature)
    if args.cmab_filter == "single-arm":
        cmab_filter = CMABFilter.from_single_arm_utility(
            env.utilities, gamma=args.gamma
        )
    elif args.cmab_filter == "marginal":
        cmab_filter = CMABFilter.from_marginal(env.utilities, gamma=args.gamma)
    else:
        cmab_filter = CMABFilter.all_active()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = args.output_dir / "training_progress.jsonl"
    progress_lines: list[str] = []

    def _progress(it: int, loss: float, log_z: float) -> None:
        progress_lines.append(json.dumps({"iter": it, "loss": loss, "log_z": log_z}))

    result = train_cmab_gfn(
        env=env,
        cmab_filter=cmab_filter,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        log_z_learning_rate=args.log_z_learning_rate,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        eval_samples=args.eval_samples,
        progress_callback=_progress,
    )
    progress_log.write_text(
        "\n".join(progress_lines) + ("\n" if progress_lines else ""),
        encoding="utf-8",
    )

    summary_path = args.output_dir / "gfn_summary.json"
    summary = {
        "config": result.config,
        "training": {
            "iters": result.history.iters,
            "losses": result.history.losses,
            "log_z_trace": result.history.log_z,
            "final_loss": (
                result.history.losses[-1] if result.history.losses else None
            ),
        },
        "evaluation": {
            "n_samples": result.evaluation.n_samples,
            "unique_terminals": result.evaluation.unique_terminals,
            "mode_share_top1": result.evaluation.mode_share_top1,
            "avg_subset_size": result.evaluation.avg_subset_size,
            "learned_log_z": result.evaluation.learned_log_z,
            "subset_counts": result.evaluation.subset_counts,
            "empirical_freqs": result.evaluation.subset_freqs,
            "target_freqs": result.evaluation.target_freqs,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Persist policy weights so the trained sampler can be replayed offline.
    import torch as _torch  # local import: torch only required for [gfn]

    weights_path = args.output_dir / "gfn_policy.pt"
    _torch.save(
        {"state_dict": result.model.state_dict(), "config": result.config},
        weights_path,
    )

    manifest_path = _manifest_path(summary_path)
    write_run_manifest(
        manifest_path,
        command="train-gfn",
        argv=_manifest_argv(args),
        started_utc=started_utc,
        status="completed",
        inputs=[],
        artifacts=[summary_path, weights_path, progress_log],
        settings=_settings_manifest(get_settings()),
        extra={
            "num_iters": args.num_iters,
            "batch_size": args.batch_size,
            "eval_samples": args.eval_samples,
            "cmab_filter": result.cmab_filter.summary(),
            "final_loss": (
                result.history.losses[-1] if result.history.losses else None
            ),
            "mode_share_top1": result.evaluation.mode_share_top1,
            "unique_terminals": result.evaluation.unique_terminals,
        },
    )

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "weights": str(weights_path),
                "manifest": str(manifest_path),
                "final_loss": (
                    result.history.losses[-1] if result.history.losses else None
                ),
                "unique_terminals": result.evaluation.unique_terminals,
                "mode_share_top1": result.evaluation.mode_share_top1,
                "active_arms": result.cmab_filter.summary()["active_tools"],
            }
        )
    )
