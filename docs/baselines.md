# Baselines and Metrics

Implemented in [`src/gpqa_cmab/metrics.py`](../src/gpqa_cmab/metrics.py) and
surfaced through `gpqa-cmab evaluate`, `gpqa-cmab baselines`, and the rendered
report.

## Subset table

`metrics.subset_table(rows)` returns one row per subset with:

- `n` — number of (question, subset) rows observed.
- `accuracy` — mean correctness.
- `avg_tokens` — mean total tokens for the subset.
- `token_savings_vs_all_four` — `1 - avg_tokens / avg_all_four_tokens`.
- `cost_per_correct_tokens` — total tokens divided by number correct.
- `utility` — average of the cost-aware utility defined in
  [cmab.md](cmab.md#reward-and-utility).

## Implemented baselines

| Name | What it measures | Function |
|---|---|---|
| Main-only | Cheapest possible policy, no subagents. | rows with `subset_id == "main_only"` in the subset table. |
| Single-subagent {A}, {B}, {C}, {D} | Whether each subagent helps in isolation. | rows with single-letter `subset_id`. |
| All-four {A,B,C,D} | High-capability, high-cost reference. | rows with `subset_id == "A,B,C,D"`. |
| Static pruning | Single hand-picked fixed subset, non-adaptive. | `metrics.static_pruning_baseline(rows, subset_id="A")`. |
| Random budget-matched | Random subsets whose average size matches the target subset. | `metrics.random_pruning_baseline(rows, target_subset_id="A,B,C,D", seeds=100)`. |
| Oracle fixed-subset | Best-utility subset after exhaustive evaluation (MVP only). | `metrics.oracle_fixed_subset(rows)`. |
| Self-consistency CoT-K | Plurality vote across K independent samples. | `gpqa-cmab run-self-consistency` → `experiments.self_consistency`. |

The combined baseline view is computed by `metrics.baseline_summary(rows)` and
embedded in `metrics_summary.json` by `reporting.write_evaluation_outputs`.
The `gpqa-cmab baselines` command writes a dedicated JSON for ad-hoc use.

## Why each baseline matters

- **Main-only** lower-bounds cost and tests whether subagents are needed at
  all.
- **Single-subagent** localizes the contribution of each subagent so that
  expensive ablations are not required.
- **All-four** is the high-cost reference that the CMAB must approach in
  utility while spending fewer tokens.
- **Static pruning** answers "could we just hard-code a cheap subset?" If
  static pruning matches CMAB, the bandit is not adding value.
- **Random budget-matched** rules out the trivial explanation that token
  savings are due to using fewer subagents on average, not to *intelligent*
  selection. If CMAB ≈ random at the same budget, the bandit is not learning.
- **Oracle fixed-subset** bounds the best a static policy could achieve on
  these specific questions. CMAB regret is computed against this reference.
- **Self-consistency** is the generic test-time-compute alternative: spend
  more tokens by sampling more, not by adding specialists. Useful for
  sanity-checking that subagent diversity, not raw token spend, drives the
  gains.

## Statistical analysis

- `metrics.bootstrap_ci(values, seed=0, samples=1000)` — paired bootstrap
  intervals for any list of per-question values.
- `metrics.mcnemar_counts(a, b)` — paired discordance counts for two methods
  evaluated on overlapping question IDs.

Sample sizes in the MVP are small. Avoid overclaiming significance; prefer
non-inferiority framing for the CMAB vs all-four comparison (see
[experiments.md](experiments.md#statistical-framing)).
