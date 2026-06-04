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

The MVP's two CMAB replays both underperform sub-agent C alone:

| Policy                       | Avg accuracy | Avg tokens | Utility |
|------------------------------|--------------|------------|---------|
| `subagent C` (static)        | 0.826        | 3 060      | 0.797   |
| `A, C` (static)              | 0.849        | 4 695      | 0.801   |
| `superarm-ts` CMAB           | 0.762        | 4 463      | 0.736   |
| `structured-cmab` CMAB       | 0.515        | 1 900      | ≈0.45   |

The Structured CMAB collapses to `main_only`: zero-initialised weights
make every subset score `σ(0) = 0.5`, so the tiny `main_only` cost
(913 tokens vs 8 419 for all-four) dominates the initial selection, the
intercept calibrates to the `main_only` accuracy (0.442), and L2
regularisation pushes the per-arm weights toward zero before they ever
get a meaningful gradient signal. Super-arm Thompson Sampling treats
all 16 arms as independent Beta posteriors and ignores the structural
sharing between e.g. `{A}` and `{A, C}`, so within 86 questions per
seed it never differentiates `{A, C}` (utility 0.801) from `{B, D}`
(utility 0.581) cleanly. See [cmab.md](cmab.md) for the historical
design.

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
