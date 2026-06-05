# Durable Telemetry Database

Goal: **research data must survive any agent misbehaviour**. Every
research-relevant action/event from every module is persisted to an
append-only event store (SQLite by default, Postgres optional) with
microsecond UTC timestamps and full payloads. The store is the *source of
truth*; the JSONL files under `artifacts/` are a grep-friendly cache that
can be rebuilt at any time from the event store.

## Why event sourcing

Mutable artefacts (`artifacts/results/*.jsonl`) are easy to overwrite by
mistake — a misbehaving coding agent re-running `smoke-test` will clobber
canonical factorial data. The telemetry DB is **append-only by
construction**: the only INSERTs are new event rows keyed by a UUID; no
UPDATE or DELETE statements exist in the production code path. Even a
catastrophic agent run cannot retroactively remove or alter prior events.

## Architecture

```
gpqa_cmab/telemetry_db/
├── __init__.py            # Public API
├── schema.py              # EventType (closed enum) + TelemetryEvent (frozen Pydantic) + DDL
├── backend.py             # TelemetryBackend Protocol (the abstraction)
├── sqlite_backend.py      # WAL-mode SQLite implementation (default, zero-dep)
├── postgres_backend.py    # psycopg-based Postgres implementation (extra)
├── config.py              # DB URL parsing + open_backend()
├── recorder.py            # TelemetryRecorder facade + process-wide default
└── reconstruct.py         # Rebuild factorial/bandit JSONL from events
```

| Layer | Responsibility |
|---|---|
| `schema.py` | Single source of truth for the event shape. `EventType` is a closed `StrEnum` — adding a new event requires a code change + schema-version bump. |
| `backend.py` | `TelemetryBackend` Protocol with `append`, `append_many`, `iter_events`, `count_events`, `close`. Both backends are interchangeable. |
| `sqlite_backend.py` | WAL journaling + `synchronous=NORMAL` for per-event durability without fsync-on-every-write latency. `INSERT OR IGNORE` makes writes idempotent on `event_uuid`. |
| `postgres_backend.py` | `psycopg` autocommit, `ON CONFLICT (event_uuid) DO NOTHING`. Lazy-imports `psycopg` so the base install stays free of native deps. |
| `recorder.py` | High-level API. `recorder.run(command=...)` opens a scoped `run_id`, emits paired `RUN_STARTED`/`RUN_FINISHED`/`RUN_FAILED`. |
| `reconstruct.py` | Read-side: rebuild factorial JSONL, bandit-replay JSONL, or run summary from the event log alone. |

## Schema

Single append-only table:

| Column | SQLite | Postgres | Notes |
|---|---|---|---|
| `id` | `INTEGER PK AUTOINC` | `BIGSERIAL PK` | Insertion order tiebreaker |
| `event_uuid` | `TEXT UNIQUE` | `UUID UNIQUE` | Idempotency key |
| `ts_utc` | `TEXT` (ISO8601) | `TIMESTAMPTZ` | Capture timestamp, µs precision |
| `run_id` | `TEXT` | `TEXT` | Groups events of one CLI invocation |
| `module` | `TEXT` | `TEXT` | Producer module, e.g. `gpqa_cmab.llm` |
| `event_type` | `TEXT` | `TEXT` | One of `EventType` |
| `schema_version` | `INTEGER` | `INTEGER` | Bumped on breaking payload shape changes |
| `payload_json` | `TEXT` | `JSONB` | Structured payload |
| `parent_event_uuid` | `TEXT` | `UUID` | Optional causal link |
| `git_sha` | `TEXT` | `TEXT` | Repo HEAD at capture |

Indexes on `ts_utc`, `run_id`, `event_type`, `module`.

## Configuration

URL resolution order (highest first):

1. Explicit argument to `open_backend(url=...)`.
2. `GPQA_TELEMETRY_DB_URL` environment variable.
3. Default: `sqlite:///artifacts/telemetry.sqlite`.

Examples:

```bash
# Default — local SQLite, always available, no network
gpqa-cmab smoke-test --mock

# Override to a different SQLite path
GPQA_TELEMETRY_DB_URL='sqlite:////absolute/path/telemetry.sqlite' \
    gpqa-cmab smoke-test --mock

# Use Postgres (requires the [telemetry-pg] extra)
uv sync --extra telemetry-pg
GPQA_TELEMETRY_DB_URL='postgresql://user:pw@host:5432/gpqa' \
    gpqa-cmab run-factorial --input data/gpqa_diamond.csv
```

## CLI

```bash
gpqa-cmab telemetry-db init             # Create tables/indexes
gpqa-cmab telemetry-db runs             # List every run with status + counts
gpqa-cmab telemetry-db verify           # Cheap integrity check
gpqa-cmab telemetry-db tail --limit 20  # Stream latest events
gpqa-cmab telemetry-db tail --run-id X --event-type llm.response
gpqa-cmab telemetry-db reconstruct summary    --run-id X
gpqa-cmab telemetry-db reconstruct factorial  --run-id X --output out.jsonl
gpqa-cmab telemetry-db reconstruct bandit     --run-id X --output out.jsonl
```

The `telemetry-db` subcommands are **read-only** and do not themselves
produce telemetry events (they are excluded from the recorder scope).

## What gets recorded today

Every CLI invocation other than `telemetry-db` is wrapped in a
`recorder.run(command=...)` scope. The following events are emitted
automatically:

| EventType | Producer | Trigger |
|---|---|---|
| `run.started`   | `cli.main` | every CLI invocation |
| `run.finished`  | `cli.main` | normal exit |
| `run.failed`    | `cli.main` | any exception |
| `llm.response`  | `telemetry.TelemetryLogger.append` | every LLM call (request + response + usage + latency + sha256 of prompt/response) |
| `artifact.written`     | `telemetry.write_jsonl` | every JSONL artefact written |
| `artifact.manifest_written` | `telemetry.write_run_manifest` | every run manifest |

Additional event types are reserved in `EventType` for future producers
(`bandit.step`, `gfn.train_iter`, `cost.budget_tick`, etc.) — these slots
are stable across schema versions so producers can be added incrementally
without breaking readers.

## Reconstruction

If `artifacts/` is wiped:

```bash
# List runs and pick the one to rebuild
gpqa-cmab telemetry-db runs

# Rebuild a factorial JSONL from the event log
gpqa-cmab telemetry-db reconstruct factorial \
    --run-id factorial-abc123 \
    --output artifacts/results/full_factorial_results.jsonl
```

The reconstructed file is byte-for-byte identical to the original up to
JSON key ordering (we serialise with `sort_keys=True`).

## Failure semantics

* **Recorder never raises.** `telemetry.py`'s `_emit_*_event` helpers
  swallow every exception so a transient DB failure cannot crash the
  research pipeline. The JSONL fallback is always written first.
* **Writes are durable on return.** Both backends commit per-event (no
  buffered batches that could be lost on `SIGKILL`).
* **Idempotent re-runs.** Re-inserting the same `event_uuid` is a no-op
  in both backends. Producers can retry safely.

## Testing

* `tests/test_telemetry_db.py` — 17 SQLite unit tests, run on every CI
  invocation. No network, no external services.
* `tests/test_telemetry_db_postgres.py` — Postgres integration tests,
  skipped unless `GPQA_TELEMETRY_PG_DSN` is set in the environment. Run
  locally with:

  ```bash
  GPQA_TELEMETRY_PG_DSN='postgresql://postgres:postgres@localhost:5432/gpqa_test' \
      uv run pytest tests/test_telemetry_db_postgres.py -v
  ```

## Schema evolution

* `SCHEMA_VERSION` in `schema.py` is bumped on every backwards-incompatible
  payload-shape change.
* The version is recorded on every row so old events stay decodable.
* The `EventType` enum is closed: adding a value requires a code change
  that the type checker will surface to every consumer.

## See also

* [ADR-0006: Durable telemetry event store](decisions/ADR-0006-telemetry-event-store.md)
* [docs/architecture.md](architecture.md) — high-level layout
* [docs/runbook.md](runbook.md) — operational procedures
