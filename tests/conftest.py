from __future__ import annotations

import json
from pathlib import Path

import pytest

# Environment variables that influence how the OpenAI-compatible client is
# constructed (provider endpoint, auth, and the chat-vs-responses-API auto
# decision). The CI runner may have some of these set globally; if a test does
# not explicitly control them it must not silently inherit the runner's values,
# otherwise behaviour (e.g. auto-enabling the Responses API when
# ``REASONING_EFFORT`` is present) leaks in and breaks otherwise-hermetic tests.
_LLM_CLIENT_ENV_VARS = (
    "REASONING_EFFORT",
    "LLM_USE_RESPONSES_API",
    "MAX_OUTPUT_TOKENS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_KEY_COOLDOWN_S",
    "OPENAI_MAX_RETRIES",
    "OPENAI_MAX_WAIT_S",
    "LLM_DEFAULT_HEADERS",
    "LLM_TIMEOUT_S",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)


@pytest.fixture(autouse=True)
def _isolate_llm_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a clean slate for LLM client-decision env vars.

    Tests that need a specific value set it explicitly via ``monkeypatch``;
    this fixture only removes any ambient values so the suite is hermetic
    regardless of the runner's environment.
    """
    for name in _LLM_CLIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "gpqa.jsonl"
    rows = [
        {
            "question_id": "q1",
            "domain": "physics",
            "question": "Mock physics?",
            "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_answer": "A",
        },
        {
            "question_id": "q2",
            "domain": "chemistry",
            "question": "Mock chemistry?",
            "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct_answer": "B",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path
