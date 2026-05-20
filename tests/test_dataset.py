from __future__ import annotations

from gpqa_cmab.dataset import load_questions, normalize_record


def test_load_questions_filters_domain(sample_jsonl):
    questions = load_questions(sample_jsonl, "physics")
    assert len(questions) == 1
    assert questions[0].question_id == "q1"
    assert questions[0].correct_answer == "A"


def test_normalize_gpqa_csv_record_shuffles_correct_answer_deterministically():
    record = {
        "Record ID": "record-1",
        "High-level domain": "Physics",
        "Question": "Q",
        "Correct Answer": "right",
        "Incorrect Answer 1": "wrong1",
        "Incorrect Answer 2": "wrong2",
        "Incorrect Answer 3": "wrong3",
    }
    first = normalize_record(record, 0, seed=7)
    second = normalize_record(record, 0, seed=7)
    assert first == second
    assert first.choices[first.correct_answer] == "right"
    assert first.domain == "physics"
