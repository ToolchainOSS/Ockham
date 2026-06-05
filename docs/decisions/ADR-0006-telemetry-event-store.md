# ADR-0006: Durable Telemetry Event Store

* Status: Accepted
* Date: 2026-06-05

## Context

The MVP persisted research data exclusively to JSONL files under
`artifacts/`. This is grep-friendly but **mutable**: any rerun of
`smoke-test` or `run-factorial` overwrites the canonical JSONL in-place,
and at least once a misbehaving coding agent has clobbered the canonical
86-question factorial that took real LLM tokens to produce
(see `/memories/repo/artifacts-gotcha.md`). Restoring from a backup tarball
is manual and not always available.

We need a storage layer where:

1. Writes are append-only by construction — no agent can retroactively
   alter or remove prior events.
2. Research artefacts (factorial JSONL, bandit replay JSONL, etc.) can
   be reconstructed from the store even if `artifacts/` is wiped.
3. The default install requires no external services or network access
   (per AGENTS.md: "Mock mode must work without API keys").
4. The same code path scales from single-machine SQLite to a shared
   Postgres instance without changing call sites.

## Decision

Add an event-sourced telemetry layer at `gpqa_cmab.telemetry_db`:

* Single `telemetry_events` table, append-only, keyed by `event_uuid`
  (`INSERT OR IGNORE` / `ON CONFLICT DO NOTHING` for idempotent retries).
* Two interchangeable backends behind a `TelemetryBackend` Protocol:
  * `SqliteBackend` — default, zero-dep, WAL-mode, per-event durable.
  * `PostgresBackend` — opt-in via `gpqa-cmab[telemetry-pg]` extra,
    autocommit, lazy import of `psycopg`.
* `TelemetryRecorder` facade with a `run(command=...)` context manager
  that emits paired `RUN_STARTED` / `RUN_FINISHED` / `RUN_FAILED`
  events.
* `EventType` is a closed `StrEnum` — adding a new event type requires
  a code change so all producers and consumers stay type-checked.
* The existing `TelemetryLogger.append()`, `write_jsonl()`, and
  `write_run_manifest()` dual-write into the active recorder if one is
  installed; the JSONL files remain as a grep-friendly cache.
* CLI subcommand `gpqa-cmab telemetry-db {init,runs,tail,verify,reconstruct}`
  for inspection and artefact reconstruction.

URL resolution: explicit arg > `GPQA_TELEMETRY_DB_URL` env > default
`sqlite:///artifacts/telemetry.sqlite`.

## Consequences

**Positive**

* Research data is now durable through any agent misbehaviour: even
  `rm -rf artifacts/results/` keeps the underlying events in the
  SQLite/Postgres store, and `gpqa-cmab telemetry-db reconstruct` can
  rebuild the JSONL artefacts.
* The default behaviour is zero-config and offline — every existing
  CLI command now writes a durable trace alongside its JSONL outputs.
* Postgres support is one env var away when running multi-node or
  shared experiments.
* The closed `EventType` enum guarantees future producers/consumers
  cannot silently disagree about the event vocabulary.

**Negative**

* Each CLI invocation now opens a SQLite connection and writes ~30+
  rows for `smoke-test`. Measured overhead on the smoke path: < 50 ms.
* `artifacts/telemetry.sqlite` grows monotonically. Gitignored, but
  may need periodic archival on production runs.
* A second persistence layer adds one more place to look during a
  debugging session. Mitigated by `telemetry-db verify` and `tail`.

**Neutral / explicitly rejected alternatives**

* *Just version-control the JSONL files.* Rejected: AGENTS.md forbids
  committing benchmark data, and per-event git commits would dominate
  the repo history.
* *Write JSONL files as append-only.* Partially mitigates the issue
  but does not survive `rm` and offers no run-id grouping or
  cross-process querying.
* *Use a full SQL ORM (SQLAlchemy).* Rejected as over-engineering for
  a single table; `sqlite3` + `psycopg` are sufficient and keep the
  base install dep-free.
* *Use an external service (e.g. OpenTelemetry collector).* Rejected:
  violates the "local reproducibility / no network" constraint.

## Related

* [docs/telemetry_db.md](../telemetry_db.md) — usage & architecture.
* `/memories/repo/artifacts-gotcha.md` — the smoke-test-clobbering
  incident that motivated this ADR.
