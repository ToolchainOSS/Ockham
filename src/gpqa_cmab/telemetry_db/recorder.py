"""High-level recorder API used by the rest of the codebase.

Every module that wants to persist an event calls into a single
:class:`TelemetryRecorder` instance. The recorder owns the backend, attaches
default metadata (run_id, git_sha, module), and exposes a small surface:

* :meth:`record` — fire a single event with arbitrary payload.
* :meth:`run` — context manager that scopes a ``run_id`` and emits matching
  ``RUN_STARTED`` / ``RUN_FINISHED`` / ``RUN_FAILED`` events.
* :meth:`bind` — return a child recorder with a fixed ``module`` so callers
  don't repeat the module name on every call.

A process-wide default recorder is available via :func:`get_recorder`. It is
created lazily on first use so importing the package never opens a database
connection.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from gpqa_cmab.telemetry_db.backend import TelemetryBackend
from gpqa_cmab.telemetry_db.config import open_backend
from gpqa_cmab.telemetry_db.schema import EventType, TelemetryEvent


class TelemetryRecorder:
    """Thread-safe facade over a :class:`TelemetryBackend`."""

    def __init__(
        self,
        backend: TelemetryBackend,
        *,
        run_id: str | None = None,
        module: str = "gpqa_cmab",
        git_sha: str | None = None,
    ) -> None:
        self._backend = backend
        self._run_id = run_id
        self._module = module
        self._git_sha = git_sha if git_sha is not None else _detect_git_sha()
        self._lock = threading.Lock()

    # -- factories ------------------------------------------------------

    def bind(self, *, module: str, run_id: str | None = None) -> TelemetryRecorder:
        """Return a child recorder with a fixed module / run_id."""
        return TelemetryRecorder(
            self._backend,
            run_id=run_id if run_id is not None else self._run_id,
            module=module,
            git_sha=self._git_sha,
        )

    # -- writes ---------------------------------------------------------

    def record(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        module: str | None = None,
        run_id: str | None = None,
        parent_event_uuid: uuid.UUID | None = None,
    ) -> TelemetryEvent:
        """Record one event. Returns the persisted :class:`TelemetryEvent`."""
        effective_run = run_id or self._run_id
        if effective_run is None:
            raise RuntimeError(
                "TelemetryRecorder.record() requires a run_id — call .run() "
                "or pass run_id=... explicitly."
            )
        event = TelemetryEvent(
            run_id=effective_run,
            module=module or self._module,
            event_type=event_type,
            payload=dict(payload or {}),
            parent_event_uuid=parent_event_uuid,
            git_sha=self._git_sha,
        )
        with self._lock:
            self._backend.append(event)
        return event

    # -- run scope ------------------------------------------------------

    @contextmanager
    def run(
        self,
        *,
        command: str,
        argv: list[str] | None = None,
        run_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Open a run scope that emits paired RUN_STARTED / RUN_FINISHED
        events. On exception, RUN_FAILED carries the exception type/message
        and is re-raised."""

        rid = run_id or _make_run_id(command)
        previous = self._run_id
        self._run_id = rid
        started_ns = time.monotonic_ns()
        self.record(
            EventType.RUN_STARTED,
            {
                "command": command,
                "argv": argv or [],
                "extra": extra or {},
                "pid": os.getpid(),
            },
        )
        try:
            yield rid
        except BaseException as exc:
            self.record(
                EventType.RUN_FAILED,
                {
                    "command": command,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "duration_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
                },
            )
            raise
        else:
            self.record(
                EventType.RUN_FINISHED,
                {
                    "command": command,
                    "duration_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
                },
            )
        finally:
            self._run_id = previous

    # -- introspection --------------------------------------------------

    @property
    def backend(self) -> TelemetryBackend:
        return self._backend

    @property
    def run_id(self) -> str | None:
        return self._run_id


# ---------------------------------------------------------------------------
# Process-wide default recorder.
# ---------------------------------------------------------------------------


_default_lock = threading.Lock()
_default_recorder: TelemetryRecorder | None = None


def get_recorder() -> TelemetryRecorder:
    """Return (or lazily create) the process-wide default recorder."""
    global _default_recorder  # noqa: PLW0603 — guarded lazy singleton
    with _default_lock:
        if _default_recorder is None:
            backend = open_backend()
            _default_recorder = TelemetryRecorder(backend)
        return _default_recorder


def get_active_recorder() -> TelemetryRecorder | None:
    """Return the currently installed recorder *without* creating one.

    Producers in low-level modules (e.g. ``telemetry.py``) use this to
    decide whether to fan their writes into the durable store. Returning
    ``None`` means "no DB sink is active, skip the dual-write" — this keeps
    unit tests free of DB side-effects unless they opt in via
    :func:`set_recorder`.
    """
    with _default_lock:
        return _default_recorder


def set_recorder(recorder: TelemetryRecorder | None) -> None:
    """Override (or reset) the process-wide default recorder.

    Pass ``None`` to clear — the next :func:`get_recorder` call will
    re-create the default.
    """
    global _default_recorder  # noqa: PLW0603 — guarded process-wide singleton
    with _default_lock:
        _default_recorder = recorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_id(command: str) -> str:
    return f"{command}-{uuid.uuid4().hex[:12]}"


def _detect_git_sha() -> str | None:
    sha = os.environ.get("GPQA_GIT_SHA")
    if sha:
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except FileNotFoundError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


__all__ = [
    "TelemetryRecorder",
    "get_active_recorder",
    "get_recorder",
    "set_recorder",
]
