"""Phase 1: the CMAB-GFN explorer.

A small Generative Flow Network (GFlowNet) trained with the Trajectory
Balance (TB) objective to sample subsets of optional subagents
``S ⊆ {A, B, C, D}`` proportionally to a strictly-positive utility-derived
reward. A CMAB pre-filter draws a γ-bounded perimeter around the action
space so the GFN does not waste flow on demonstrably dead branches.

This module is intentionally **isolated** from the LLM/replay pipeline: it
imports nothing from ``gpqa_cmab.llm`` or ``gpqa_cmab.experiments`` and
only depends on PyTorch at call time. Tests skip cleanly when ``torch``
is not installed (see ``pyproject.toml`` extra ``gfn``).
"""

from __future__ import annotations

from gpqa_cmab.gfn.cmab_filter import (
    CMABFilter,
    marginal_contributions,
    single_arm_utilities,
)
from gpqa_cmab.gfn.empirical import (
    EMPIRICAL_UTILITIES,
    TOOLS,
    subset_to_id,
    utilities_table,
)
from gpqa_cmab.gfn.environment import SubagentEnvironment

__all__ = [
    "CMABFilter",
    "EMPIRICAL_UTILITIES",
    "SubagentEnvironment",
    "TOOLS",
    "marginal_contributions",
    "single_arm_utilities",
    "subset_to_id",
    "utilities_table",
]
