# Dataset Contract

## Supported formats

`gpqa_cmab.dataset.load_questions(path, domain, max_questions)` accepts:

- **JSONL** — one normalized record per line (the canonical format).
- **CSV** — a `question_id,domain,question,A,B,C,D,correct_answer` header,
  where `A..D` carry the choice text and `correct_answer` is `A`, `B`, `C`,
  or `D`.

The CSV reader is convenient for the bundled `data/gpqa_diamond.csv` sample;
production runs should prefer JSONL.

## Normalized record schema

Every record produced by the loader conforms to `GPQAQuestion` in
[`src/gpqa_cmab/schemas.py`](../src/gpqa_cmab/schemas.py):

```json
{
  "question_id": "string",
  "domain": "physics",
  "question": "string",
  "choices": {"A": "string", "B": "string", "C": "string", "D": "string"},
  "correct_answer": "A|B|C|D"
}
```

The domain is lowercased on load. Filtering happens before `max_questions` is
applied so caps are deterministic.

## Privacy and leakage

- Question text and gold answers are not echoed to reports. The Markdown report
  uses question IDs and aggregate counts only.
- Datasets are not committed to the repository. `data/.gitignore` keeps the
  folder around; the contents must be supplied by the operator.
- Mock mode embeds the gold answer in the prompt under
  `MOCK_CORRECT_ANSWER=...` so that the mock provider can produce
  schema-correct outputs without an LLM. This token is unique to mock prompts
  and must never appear in production prompts. See
  [ADR-0005](decisions/ADR-0005-mock-provider-default.md).

## Extending to other domains

To run chemistry or biology, drop the corresponding records into the input file
with `domain` set accordingly. The pipeline does not bake in any physics-only
assumptions outside of the prompts in `prompts/`, which are versioned and can
be swapped or extended per domain. Prompt versions are recorded in every
`FactorialResult` row.
