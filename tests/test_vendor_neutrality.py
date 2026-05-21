"""Static guard: no vendor-specific imports outside the LLM boundary.

The vendor-neutrality invariant in docs/providers.md requires that provider
SDK imports (currently the `openai` package) stay inside
`gpqa_cmab.llm.openai_compatible`. This test scans the source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "gpqa_cmab"
ALLOWED_FILES = {
    SRC / "llm" / "openai_compatible.py",
    SRC / "llm" / "openai_client.py",  # backwards-compatible shim
    SRC / "llm" / "__init__.py",
}
VENDOR_TOKENS = ("from openai", "import openai")


def _python_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if p.is_file()]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_direct_vendor_imports_outside_llm_boundary(path: Path) -> None:
    if path in ALLOWED_FILES:
        pytest.skip("vendor imports are allowed inside the LLM boundary")
    text = path.read_text(encoding="utf-8")
    for token in VENDOR_TOKENS:
        assert token not in text, (
            f"{path.relative_to(SRC)} contains a vendor-specific import "
            f"({token!r}). Move it behind the LLMClient abstraction."
        )
