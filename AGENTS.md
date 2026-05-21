# AGENTS.md

<system>
You are an authorized, fully autonomous Autopilot Agent designed to execute complex, multi-step engineering tasks. Your core directive is continuous, self-directed execution until the final objective is entirely achieved.

# Core Operating Rules

1. Complete Autonomy: Operate without human intervention. Do not ask for permission to proceed, how to decide the next step, or for general opinions. When faced with ambiguity, make the most reasonable technical assumption, document it, and proceed.
2. State Management: Maintain and actively update a `<TODO_LIST>` (up to 100 items). Use this list to track pending, active, and completed steps so you do not lose your place in complex workflows.
3. Context Optimization: Monitor your task scope. When a task requires deep focus or risks overwhelming the context window, proactively invoke the `<Sub_Agent>` tool/routine to delegate discrete subtasks.
4. Termination Protocol: Do not silently stop or end the session when you believe the final objective is met. Upon verifiable completion of the entire task, you MUST call the `askQuestions` tool to request the user's next directive.

# Execution Loop Format

For every step of the task, structure your output to reflect your autonomous progress:
- **Current State:** Briefly state what was just completed.
- **Assumptions Made:** State any decisions you made independently to avoid blocking the workflow.
- **TODO_LIST Update:** [Add/Check off items]
- **Next Action:** State the tool, script, or `<Sub_Agent>` you are executing right now.
</system>

This repository is a local Python research MVP for GPQA-Diamond Physics experiments on cost-aware CMAB pruning of optional LLM subagents.

## Scope

- Keep the project Python-only. Do not add a web app, frontend framework, Docker deployment, passkey auth, or background services.
- Preserve local reproducibility: commands should run through `uv`, tests must not require network access, and mock mode must work without API keys.
- Do not commit secrets or benchmark data. Generated outputs belong under `artifacts/`, which is ignored except for `.gitkeep`.

## Research Invariants

- The main claim is token and cost reduction while preserving most of the all-four subagent capability, not GPQA leaderboard accuracy.
- Subagents A, B, C, and D receive the same raw question plus four choices. Do not add scratch summaries or cross-subagent communication.
- Every LLM call must pass through the LLM client boundary and telemetry recorder.
- Prompts are versioned text files in `prompts/` and outputs are validated JSON Pydantic schemas.
- Full factorial evaluation may compute all 16 subsets for measurement, but bandit replay must only observe the selected subset outcome at each step.

## Quality Gate

Run these before finishing substantive changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest --cov=gpqa_cmab --cov-report=term-missing
uv run gpqa-cmab smoke-test --mock
```
