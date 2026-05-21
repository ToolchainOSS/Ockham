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

$$\text{score}(S) = \sigma(w^\top \phi(S)) - \lambda_{\text{token}} \cdot \tilde{T}(S) - \lambda_{\text{call}} \cdot |S| + \frac{c}{\sqrt{1 + n_S}}$$

The last term is a Thompson-sampling-style uncertainty bonus that shrinks as a
subset is observed more often.

Why this matters: an observation of `{A, C, D}` updates the weights for `A`,
`C`, `D`, and the pairwise interactions `A·C`, `A·D`, `C·D`. Untested subsets
inherit predictions through shared features. This is the CMAB rationale that
generalizes beyond the toy 16-subset case.

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
