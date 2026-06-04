"""Canonical 86-question MVP aggregates per subset.

These numbers come from the original real-LLM ``run-factorial`` artifact
(``artifacts/results/metrics_summary.json``, n=86 GPQA-Diamond Physics)
and are mirrored here so the offline ``benchmark-cmab`` command stays
reproducible even if the gitignored artifact gets overwritten or rebuilt
under a different mock seed.

Source of truth: ``artifacts/reports/mvp_report.md`` + the original
``metrics_summary.json`` written by the MVP run with manifest
``sha256: bc8f6f6c1d806beeb694cb3bde8b0442702ddb6f34ab0a2516b3a43a3abb03da``.
"""

from __future__ import annotations

# (subset_id, accuracy, avg_tokens) at n=86 questions per subset.
MVP_SUBSET_AGGREGATES: tuple[tuple[str, float, float], ...] = (
    ("main_only", 0.442, 913.2),
    ("A", 0.767, 2493.1),
    ("B", 0.570, 3025.6),
    ("C", 0.826, 3060.2),
    ("D", 0.593, 2523.2),
    ("A,B", 0.767, 4598.9),
    ("A,C", 0.849, 4695.3),
    ("A,D", 0.756, 4150.4),
    ("B,C", 0.814, 5159.3),
    ("B,D", 0.628, 4610.7),
    ("C,D", 0.756, 4703.3),
    ("A,B,C", 0.837, 6784.4),
    ("A,B,D", 0.744, 6239.8),
    ("A,C,D", 0.849, 6332.4),
    ("B,C,D", 0.791, 6795.4),
    ("A,B,C,D", 0.826, 8419.5),
)

N_MVP_QUESTIONS: int = 86
