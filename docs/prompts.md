# Prompts and JSON Contracts

## Layout

Prompts are plain-text files under [`prompts/`](../prompts/), named with a
version suffix:

```text
prompts/
├── main_integrator_v1.txt
├── subagent_A_specialist_v1.txt
├── subagent_B_reference_v1.txt
├── subagent_C_computation_v1.txt
├── subagent_D_verifier_v1.txt
└── self_consistency_v1.txt
```

Versioning rule: any change that alters the *meaning* or the *output contract*
of a prompt requires a new `..._vN.txt` file. Backwards-compatible wording
tweaks may stay on the same version, but be conservative — every
`FactorialResult` records the prompt version used for the main integrator and
each subagent, so version churn is recoverable.

`gpqa_cmab.prompts.load_prompt(name)` reads the prompt file, and
`prompt_version(name)` returns the trailing `vN` token. Both are pure
functions over `prompts/` and have no I/O beyond reading the text file.

## JSON output contracts

Every agent must return a single JSON object that validates against a Pydantic
model in [`src/gpqa_cmab/schemas.py`](../src/gpqa_cmab/schemas.py):

| Agent | Pydantic model |
|---|---|
| Main integrator | `MainIntegratorOutput` |
| Subagent A (specialist) | `SubagentAReport` |
| Subagent B (reference) | `SubagentBReport` |
| Subagent C (computation) | `SubagentCReport` |
| Subagent D (verifier) | `SubagentDReport` |
| Self-consistency sample | `SelfConsistencyOutput` |

The schemas are the source of truth. Prompts must instruct the model to
produce exactly those fields. The shared helper
`gpqa_cmab.json_utils.complete_validated()` validates and retries malformed
output — see [telemetry.md](telemetry.md#json-validation-and-retries).

## Hidden chain-of-thought

The brief explicitly disallows hidden chain-of-thought in final logs. The
`rationale_summary` field is a *short* explanation only. If a real model
produces a long internal trace, do not store it in telemetry or reports.

## Adding a new subagent

1. Add a `SubagentXReport` Pydantic model in `schemas.py`.
2. Create `prompts/subagent_X_<role>_v1.txt`.
3. Register the prompt + schema in `gpqa_cmab.agents.subagents` (`SCHEMAS` and
   `PROMPTS` dicts).
4. Update [`gpqa_cmab/subsets.py`](../src/gpqa_cmab/subsets.py) — note that the
   subset space grows to `2^n`, and the structured CMAB feature vector in
   [`bandits/structured_cmab.py`](../src/gpqa_cmab/bandits/structured_cmab.py)
   needs new singletons and pair interactions.
5. Add mock payloads in [`llm/mock.py`](../src/gpqa_cmab/llm/mock.py).
6. Extend tests in `tests/test_subsets.py` and `tests/test_schemas.py`.
