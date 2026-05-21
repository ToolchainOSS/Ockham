from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable
from pathlib import Path

from gpqa_cmab.schemas import GPQAQuestion

CHOICES = ("A", "B", "C", "D")


def load_questions(
    path: Path,
    domain: str = "physics",
    max_questions: int | None = None,
    seed: int = 0,
) -> list[GPQAQuestion]:
    records = list(_read_records(path))
    questions = [
        normalize_record(record, index, seed) for index, record in enumerate(records)
    ]
    filtered = [q for q in questions if q.domain == domain.lower()]
    return filtered[:max_questions] if max_questions else filtered


def _read_records(path: Path) -> Iterable[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _first(record: dict[str, str], *keys: str, default: str = "") -> str:
    """Return the first non-empty value among ``keys`` in ``record``."""
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return default


def normalize_record(record: dict[str, str], index: int, seed: int = 0) -> GPQAQuestion:
    if "choices" in record:
        choices = record["choices"]
        if isinstance(choices, str):
            choices = json.loads(choices)
        return GPQAQuestion(
            question_id=_first(record, "question_id", "id", default=str(index)),
            domain=_first(record, "domain", "High-level domain").lower(),
            question=str(record["question"]),
            choices=choices,
            correct_answer=str(record["correct_answer"]).upper(),
        )

    question = _first(record, "Question", "question")
    correct = _first(record, "Correct Answer", "correct_answer")
    incorrect = [
        record.get("Incorrect Answer 1", ""),
        record.get("Incorrect Answer 2", ""),
        record.get("Incorrect Answer 3", ""),
    ]
    rng = random.Random(f"{seed}:{record.get('Record ID', index)}")
    labeled = [("correct", correct), *[("incorrect", item) for item in incorrect]]
    rng.shuffle(labeled)
    choices = {label: text for label, (_, text) in zip(CHOICES, labeled, strict=True)}
    correct_answer = next(
        label
        for label, (kind, _) in zip(CHOICES, labeled, strict=True)
        if kind == "correct"
    )
    return GPQAQuestion(
        question_id=_first(record, "Record ID", "question_id", default=str(index)),
        domain=_first(record, "High-level domain", "domain").lower(),
        question=question,
        choices=choices,
        correct_answer=correct_answer,
    )


def question_context(question: GPQAQuestion) -> str:
    choices = "\n".join(f"{key}. {question.choices[key]}" for key in CHOICES)
    return f"Question:\n{question.question}\n\nChoices:\n{choices}"
