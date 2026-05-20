from __future__ import annotations

import json
from pathlib import Path

import pytest


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
