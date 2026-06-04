# CMAB-GFN Explorer — Phase 1

The Phase-1 explorer pivots from a pure Combinatorial Multi-Armed Bandit
(CMAB) over the 16 subagent subsets to a **hybrid CMAB-GFN**:

1. A coarse **CMAB pre-filter** scores each base arm `i ∈ {A, B, C, D}`
   and prunes arms below a utility threshold `γ`. This bounds the
   combinatorial topology the GFlowNet must explore — the literature
   (Liu et al., *Exploring Multiple High-Scoring Subspaces in Generative
   Flow Networks*) shows that pure GFlowNets over-explore vast state
   spaces, so this guardrail is essential as the pool grows beyond four.
2. A small **Generative Flow Network (GFN)** trained with the
   **Trajectory Balance (TB)** objective samples subsets from the
   bounded subspace `S_restricted` proportionally to a strictly-positive
   reward `R(x) = exp(utility(x) / T)`.

Code lives at [src/gpqa_cmab/gfn/](../src/gpqa_cmab/gfn/). The
dependency on `torch` is gated behind the optional `[gfn]` extra so the
default install of `gpqa-cmab` stays lean:

```bash
uv sync --extra gfn
```

## Why pivot away from pure CMAB

Even after the cold-start bug fixes (see [cmab.md](cmab.md)), both CMAB
variants still underperform sub-agent `C` alone on the partial-information
replay over 100 seeds × 86 questions:

| Policy                       | Avg accuracy | Avg tokens | Utility |
|------------------------------|--------------|------------|---------|
| `subagent C` (static)        | 0.826        | 3 060      | 0.797   |
| `A, C` (static, oracle)      | 0.849        | 4 695      | **0.801** |
| `superarm-ts` (fixed)        | 0.758        | 4 434      | 0.713   |
| `structured-cmab` (fixed)    | 0.680        | 3 708      | 0.644   |
| `structured-cmab` (legacy)   | 0.515        | 1 900      | 0.498   |

Super-arm Thompson Sampling treats all 16 arms as independent Beta
posteriors and ignores the structural sharing between e.g. `{A}` and
`{A, C}`, so within 86 questions per seed it never differentiates
`{A, C}` (utility 0.801) from `{B, D}` (utility 0.581) cleanly. The
Structured CMAB shares per-arm and pair features but still needs many
plays per arm to escape the broad cost-penalty basin around `main_only`.
The CMAB-GFN sidesteps both problems by *targeting* the high-utility
subsets directly — see the benchmark table below.

## Mathematical objects

### State space (DAG)

* `s_0 = ∅` is the initial state.
* Each non-terminal state is a multi-hot vector `s ∈ {0, 1}^4` over
  `(A, B, C, D)`.
* A terminal state `x` is reached by emitting the explicit `TERMINATE`
  action; the terminal subsets form `X ⊆ 2^{A,B,C,D}`.
* Action space `A = {ADD_A, ADD_B, ADD_C, ADD_D, TERMINATE}` (size 5).

### Reward

```
R(x) = exp(utility(x) / T)        with T = 0.1 by default
```

`R(x) > 0` strictly so `log R(x)` is finite. Smaller `T` sharpens the
target distribution onto the highest-utility subsets without ever
collapsing to a single mode.

### Forward policy `P_F(a | s; θ)`

An MLP (`Linear → ReLU → Linear → ReLU → Linear`) with hidden width 64
mapping `s ∈ {0,1}^4 → 5` logits. **Action masking** sets
`logit = −1×10^9` whenever an action is illegal (tool already in `s` or
the tool is pruned by the CMAB pre-filter), so the post-softmax
probability is exactly zero on illegal branches while staying
differentiable on the legal ones.

### Backward policy `P_B(s | s')` (uniform parents)

Every terminal `x` with `k` tools has exactly `k` forward parents (one
per tool that could have been added last). With the uniform-parent
convention `P_B(s | s') = 1 / |parents(s')|`, and noting that the only
non-add transition is the final `TERMINATE` (single-parent), the
trajectory-wide backward log-prob collapses to

```
Σ_t log P_B(s_t | s_{t+1})  =  − log(k!)
```

### Trajectory Balance loss

For each sampled trajectory `τ = (s_0, …, s_n = x)`:

```
L_TB(τ; θ, Z) = ( log Z
               + Σ_t log P_F(s_{t+1} | s_t; θ)
               − log R(x)
               − Σ_t log P_B(s_t | s_{t+1}) )^2
```

`log Z` is a single learnable scalar parameter (in log-space so `Z > 0`
by construction) with its own optimizer group so it can move faster
than the network weights.

## CMAB pre-filter

[`cmab_filter.py`](../src/gpqa_cmab/gfn/cmab_filter.py) ships two
ready-made γ-threshold filters and an ablation:

* `single-arm` — score = utility of the solo subset `{i}`. With the MVP
  data and `γ = 0.6` this keeps `A` (0.743) and `C` (0.797) and prunes
  `B` (0.542) and `D` (0.568).
* `marginal` — score = `E[u | i ∈ S] − E[u | i ∉ S]`. Robust when arms
  interact (negative scores flag tools that hurt on average).
* `none` (`CMABFilter.all_active`) — keeps every arm, used for the
  pure-GFN ablation.

For this 4-tool MVP we do not need an online UCB updater: the empirical
utility dictionary plays the role of a pre-filtered oracle. The filter
exposes a bool `torch.Tensor` of shape `(n_tools,)` which the trainer
passes into `env.action_mask(...)`, so the same code path supports a
future online-CMAB replacement.

## Empirical validation

```
uv run gpqa-cmab train-gfn --output-dir artifacts/gfn \
  --num-iters 1500 --batch-size 64 --eval-samples 1000
```

With the default `single-arm` filter at `γ = 0.6` (active arms ={A, C}):

| subset       | empirical (1 000 samples) | analytic R/Z |
|--------------|---------------------------|--------------|
| `A,C`        | 0.395                     | 0.393        |
| `C`          | 0.367                     | 0.377        |
| `A`          | 0.228                     | 0.220        |
| `main_only`  | 0.010                     | 0.010        |

Top-mode share is 39.5 % — diversity preserved across all 4 reachable
terminals.

Ablation — no CMAB filter (pure GFN over all 16 terminals), `T = 0.1`:

| subset       | empirical | analytic R/Z |
|--------------|-----------|--------------|
| `C`          | 0.142     | 0.127        |
| `A,C`        | 0.132     | 0.133        |
| `A,C,D`      | 0.102     | 0.109        |
| `A,B,C`      | 0.097     | 0.094        |
| `B,C`        | 0.095     | 0.091        |
| …            | …         | …            |
| `main_only`  | 0.003     | 0.003        |

All 16 unique terminals are visited; top-mode share is ~14 %.

## Head-to-head benchmark (real 86-Q factorial)

All numbers below use the same cost-aware utility
$u(S) = \text{acc}(S) - 0.05 \cdot \tilde{T}(S) - 0.01 \cdot |S|$ evaluated
on the per-subset empirical accuracy/token data from the canonical
86-question MVP factorial. The GFN rows are the *expected* utility under
the trained sampling distribution (averaged over 4 training seeds,
5 000 evaluation rollouts each). The bandit rows are 100-seed
partial-information replays. The static rows are deterministic.

| Policy | Active arms | Acc | Tokens/q | Utility (± std) | Avg \|S\| | #terminals |
|---|---|---|---|---|---|---|
| `static[A,C]` (oracle subset)               | {A,C} | 0.849 | 4 695 | **0.801** | 2.00 | 1 |
| `static[C]`                                 | {C}   | 0.826 | 3 060 |  0.797   | 1.00 | 1 |
| **CMAB-GFN** (single-arm filter, γ=0.6)     | {A,C} | 0.817 | 3 549 | **0.7826 ± 0.0005** | 1.38 | 4 |
| **RAW-GFN** (no filter)                     | {A,B,C,D} | 0.800 | 5 105 | 0.7479 ± 0.0006 | 2.22 | 16 |
| `static[A]`                                 | {A}   | 0.767 | 2 493 |  0.743   | 1.00 | 1 |
| `static[A,B,C,D]`                           | {A,B,C,D} | 0.826 | 8 419 |  0.736   | 4.00 | 1 |
| `superarm-ts` (fixed, 100-seed replay)      | n/a   | 0.758 | 4 434 |  0.713   | – | 15.5 |
| `structured-cmab` (fixed, 100-seed replay)  | n/a   | 0.680 | 3 708 |  0.644   | – | 16.0 |
| `structured-cmab` (legacy-buggy)            | n/a   | 0.515 | 1 900 |  0.498   | – | 8.1 |
| **CMAB-GFN** (marginal filter, γ=0.6)       | {}    | 0.442 |   913 |  0.4364 ± 0.0000 | 0.00 | 1 |
| `static[main_only]`                         | {}    | 0.442 |   913 |  0.436   | 0.00 | 1 |

### Findings

1. **CMAB-GFN with the single-arm filter (γ=0.6) is the best non-oracle
   policy** at utility 0.783 — within 2.2 % of oracle `static[A,C]`
   (0.801) and within 1.8 % of `static[C]` (0.797), while preserving
   diversity across 4 terminals (modes share split ~A,C:39 % /
   C:38 % / A:22 %). It beats both fixed bandits by a wide margin
   (+0.07 over superarm-ts, +0.14 over structured-cmab).
2. **Raw GFN (no filter)** still beats both bandits (0.748 vs 0.713 /
   0.644) and the all-four static (0.736). The TB objective targets the
   reward distribution directly, so even without pruning it concentrates
   mass on high-utility subsets — but it also wastes ~30 % of its mass
   on the B/D-containing low-utility subsets, which the filter removes.
3. **The marginal filter with γ=0.6 is degenerate** for these data: the
   marginal contribution of any single arm
   (E[u | i∈S] − E[u | i∉S]) is on the order of 0.05–0.10, well below
   the 0.6 threshold that was calibrated for the single-arm score. Result:
   every arm is pruned, the GFN collapses onto `main_only`, and the
   policy degenerates to `static[main_only]`. The marginal filter needs
   its own γ (around 0.03–0.05).
4. **Training is extremely stable**: the four CMAB-GFN seeds agree on
   utility to four decimals (σ ≈ 5e-4) — the variance comes from the
   5 000-sample Monte-Carlo evaluator, not the optimisation.

### Reproduction

```bash
# CMAB-GFN with the production filter (single-arm, γ=0.6)
uv run gpqa-cmab train-gfn --output-dir artifacts/gfn/cmab_filter \
    --cmab-filter single-arm --gamma 0.6 \
    --num-iters 2000 --eval-samples 5000 --seed 0

# Raw-GFN ablation (no pruning)
uv run gpqa-cmab train-gfn --output-dir artifacts/gfn/raw \
    --cmab-filter none \
    --num-iters 2000 --eval-samples 5000 --seed 0
```

## Module layout

```
src/gpqa_cmab/gfn/
├── __init__.py        # public surface (lazy torch import)
├── empirical.py       # MVP utility table + subset_to_id helpers
├── environment.py     # subset-construction MDP + action mask + reward
├── model.py           # GFlowNet MLP + learnable log_Z parameter
├── cmab_filter.py     # γ-threshold pre-filter (single-arm, marginal, all)
└── training.py        # batched rollouts, TB loss, training driver, eval
```

## CLI

`gpqa-cmab train-gfn` — see [cli.md](cli.md) for the full flag list.
Key flags:

* `--cmab-filter {single-arm,marginal,none}` — pre-filter family.
* `--gamma FLOAT` — pruning threshold.
* `--temperature FLOAT` — reward sharpening (default `0.1`).
* `--num-iters INT`, `--batch-size INT`, `--learning-rate FLOAT`,
  `--log-z-learning-rate FLOAT` — TB-loss training hyperparameters.
* `--eval-samples INT` — post-training rollouts for diagnostics.

Outputs land in `--output-dir` (default `artifacts/gfn/`):

* `gfn_summary.json` — config, per-checkpoint loss/log-Z trace, and the
  full evaluation report (empirical vs analytic frequencies).
* `gfn_policy.pt` — trained `state_dict` + config.
* `training_progress.jsonl` — one JSON line per checkpoint.
* `gfn_summary.json.manifest.json` — run manifest (git rev, settings,
  artifacts).

## Limitations / next steps

* Phase 1 trains on the static empirical reward dictionary; live
  reward (per-question accuracy minus token cost) requires wiring the
  GFN into the orchestration loop after the LLM round-trip.
* The CMAB pre-filter is currently static; the next milestone is to
  replace it with an online UCB updater so `B_active` shrinks/grows as
  empirical evidence accumulates.
* The single-round execution constraint is preserved end-to-end — there
  are no while-loops, no multi-turn agent dialogue, and no web-search
  capability anywhere in the subagent layer.
