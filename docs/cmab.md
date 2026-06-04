# Bandits and CMAB Design

## Why a bandit at all?

The brief is explicit: with four optional subagents, `2^4 = 16` subsets is
trivially enumerable. The MVP evaluates all 16 for measurement. The CMAB is a
*deployment proxy* — it learns under partial information so that the
methodology generalizes to systems with many subagents where exhaustive
evaluation is infeasible.

## Reward and utility

Primary outcome:

$$Y = \mathbb{1}[\text{final\_answer} = \text{correct\_answer}]$$

Cost-aware utility:

$$u(S) = Y - \lambda_{\text{token}} \cdot \tilde{T}(S) - \lambda_{\text{call}} \cdot |S|$$

where $\tilde{T}(S) = T(S) / \bar{T}_{ABCD}$ is the subset's total tokens
normalized against the average all-four token count. Defaults
$\lambda_{\text{token}} = 0.05$, $\lambda_{\text{call}} = 0.01$ are
configurable via `LAMBDA_TOKEN` / `LAMBDA_CALL`.

## Subset encoding

`subsets.subset_id(tuple)` returns deterministic IDs:

- Empty tuple → `"main_only"`.
- Otherwise sorted letters joined by commas: `"A,C,D"`.

`subsets.all_subsets()` enumerates all 16 tuples in a fixed order. Both
helpers are pure and unit-tested.

## Super-arm Thompson Sampling

`gpqa_cmab.bandits.superarm_ts.SuperArmThompsonSampler` is the simplest CMAB
baseline. Each of the 16 subsets is a separate arm with a Beta-Bernoulli
posterior:

$$\theta_S \sim \text{Beta}(\alpha_S, \beta_S)$$

Selection rule:

$$\text{score}(S) = \theta_S - \lambda_{\text{token}} \cdot \tilde{T}(S) - \lambda_{\text{call}} \cdot |S|$$

The sampler picks `argmax score(S)`. Updates increment `success` or `failure`
counts based on observed correctness.

Limitations: no information sharing across arms. Sample efficiency degrades as
the subset space grows.

## Structured CMAB

`gpqa_cmab.bandits.structured_cmab.StructuredCMAB` predicts the success
probability of a subset from a feature vector

$$\phi(S) = [\,1,\ A,\ B,\ C,\ D,\ A\cdot B,\ A\cdot C,\ A\cdot D,\ B\cdot C,\ B\cdot D,\ C\cdot D,\ |S|,\ \tilde{T}(S)\,]$$

with a logistic model trained online by SGD with L2 regularization. Selection
uses

$$\text{score}(S) = \sigma(w^\top \phi(S)) - \lambda_{\text{token}} \cdot \tilde{T}(S) - \lambda_{\text{call}} \cdot |S| + c \cdot \sqrt{\frac{\log(1 + t)}{1 + n_S}}$$

The last term is a UCB1-style uncertainty bonus that grows with total plays
`t` while shrinking with arm-specific plays `n_S`.

Why this matters: an observation of `{A, C, D}` updates the weights for `A`,
`C`, `D`, and the pairwise interactions `A·C`, `A·D`, `C·D`. Untested subsets
inherit predictions through shared features. This is the CMAB rationale that
generalizes beyond the toy 16-subset case.

### Cold-start bug fix (2026 refit)

The first cut of `StructuredCMAB` shipped with three coupled cold-start bugs
that drove it to collapse onto `main_only` within ~20 steps (offline
benchmark: utility 0.50 vs the static-`C` baseline at 0.80):

1. **Pessimistic init.** Zero weights → $\sigma(0) = 0.5$ on step 1, so the
   cost penalty $-\lambda_{\text{token}} \cdot \tilde{T}(S)$ chose the
   cheapest arm (`main_only`) every time.
2. **L2-shrunk intercept.** A single wrong `main_only` step pulled the
   intercept negative, dragging every subset's score down by the same amount
   — `main_only`'s cost advantage persisted.
3. **Bonus too small.** The original `0.1/√(1+n)` bonus was the same order
   of magnitude as the cost penalty.

Fixes (defaults; legacy behaviour available for ablation):

| Knob | Legacy (buggy) | Fixed (default) |
|---|---|---|
| `prior_accuracy` | 0.5 (implicit) | 0.7 (warm-start intercept) |
| `shrink_intercept` | `True` (implicit) | `False` |
| `uncertainty` | 0.1 | 0.3 |
| `bonus_form` | `inv_sqrt_n` | `ucb1` |

Set the legacy values explicitly to reproduce the pre-fix collapse for
ablation. See [test_bandits.py](../tests/test_bandits.py) for
regression coverage.

## SuperArm-TS prior refit (2026)

The original `alpha0 = beta0 = 1` (Beta(1,1) uniform) prior was too flat
relative to the sample budget (~5 plays per arm per seed at 86 steps over 16
arms) and the TS posterior never sharpened enough to differentiate good
super-arms cleanly. The new default `Beta(alpha0=3, beta0=2)` (mean 0.6, ESS
5) anchors initial picks to a plausible accuracy band. Legacy uniform prior
remains available via `alpha0=1, beta0=1`.

## Offline benchmark on the empirical surface

`gpqa-cmab benchmark-cmab` simulates many seeds of each bandit policy on a
Bernoulli environment built from the per-subset aggregates of the 86-question
MVP factorial (baked into
`src/gpqa_cmab/experiments/mvp_aggregates.py`). This decouples
algorithm-quality measurement from the cost of re-running real LLM calls.

Result on the canonical 86Q surface (500 seeds × 86 steps, λ_token=0.05,
λ_call=0.01):

| Policy | Accuracy | Avg tokens | Utility | Unique subsets |
|---|---|---|---|---|
| `structured-cmab (fixed)`        | 0.691 | 3 774 | **0.654** | 16.00 |
| `structured-cmab (legacy-buggy)` | 0.518 | 1 879 | 0.502 | 7.83 |
| `superarm-ts (fixed)`            | 0.757 | 4 415 | **0.713** | 15.58 |
| `superarm-ts (legacy-flat-prior)`| 0.764 | 4 513 | 0.718 | 15.46 |
| `static[main_only]`              | 0.442 | 913   | 0.437 | 1 |
| `static[A]`                      | 0.768 | 2 493 | 0.743 | 1 |
| `static[C]`                      | 0.829 | 3 060 | **0.801** | 1 |
| `static[A,C]`                    | 0.852 | 4 695 | 0.805 | 1 |
| `static[A,B,C,D]`                | 0.829 | 8 419 | 0.739 | 1 |

Takeaways:

1. The cold-start fixes recover the structured CMAB from catastrophic
   collapse (utility 0.50 → 0.65, late-window `main_only` rate 88% → 29%).
2. SuperArm-TS is nearly insensitive to the prior change — its data-driven
   posterior dominates after a handful of plays.
3. Even fixed, **both bandits still lose to static-`C` alone** because they
   can't internalise the strong synergy between `A` and `C` from independent
   per-arm/pairwise updates in 86 steps. This sample-efficiency floor is
   exactly what motivates the [Phase-1 CMAB-GFN explorer](gfn.md).

## Partial-information replay protocol

`gpqa_cmab.experiments.replay.replay_bandit` simulates deployment on the
factorial table while *only* observing the chosen subset's outcome at each
step. The invariant is enforced by indexing
`by_question[question_id][selected_subset_id]` — the learner never sees rows
for unselected subsets.

For each seed and policy:

1. Shuffle the question order with the seeded RNG.
2. Per step, ask the learner for the next subset.
3. Look up the cached `FactorialResult` for that (question, subset).
4. Update the learner with `(subset, correct[, total_tokens])`.
5. Record a `BanditStep` row with cumulative utility and the count of unique
   subsets explored so far.

The output JSONL is consumed by `reporting.write_report` to compute average
accuracy, average tokens, final cumulative utility, unique subsets explored,
and regret versus the oracle fixed-subset reference.

## Oracle fixed-subset reference

After full factorial evaluation, the subset with the highest utility is the
"oracle" reference. It is *not* a deployable policy because it requires
exhaustive evaluation, but it bounds what a perfect static policy could
achieve on the observed data. Reports label it
`oracle_fixed_subset_mvp_only`.

## Extending the CMAB

If you add a fifth subagent E:

- Add singleton feature `E` and pair features `A·E, B·E, C·E, D·E` to
  `bandits/structured_cmab.FEATURES` and `features()`.
- Subset space grows to 32; `subsets.all_subsets()` already handles this if
  `SUBAGENTS` is updated.
- Consider reducing the L2 regularization or warm-starting the weights from
  the four-subagent run.
