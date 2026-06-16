"""Configuration parsing for the telemetry DB layer.

Resolution order for the DB URL (highest priority first):

1. Explicit argument to :func:`resolve_database_url` / :func:`open_backend`.
2. ``GPQA_TELEMETRY_DB_URL`` environment variable.
3. Default: ``sqlite://artifacts/telemetry.sqlite`` (always available, no
   network, no external services).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from gpqa_cmab.telemetry_db.backend import BackendKind, TelemetryBackend

_ENV_VAR: Final[str] = "GPQA_TELEMETRY_DB_URL"
_DEFAULT_SQLITE: Final[str] = "sqlite:///artifacts/telemetry.sqlite"


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Parsed, validated connection target."""

    kind: BackendKind
    raw_url: str
    sqlite_path: Path | None = None
    postgres_dsn: str | None = None


def resolve_database_url(explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    return os.environ.get(_ENV_VAR, _DEFAULT_SQLITE)


def parse_database_url(url: str) -> DatabaseConfig:
    """Parse a DB URL into a :class:`DatabaseConfig`.

    Supported schemes:

    * ``sqlite:///relative/path.db`` or ``sqlite:////absolute/path.db``
    * ``postgresql://user:pass@host:5432/dbname``
    * ``postgres://...`` (alias accepted, normalised to ``postgresql://``)
    """

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "sqlite":
        path_str = parsed.path
        if not path_str:
            raise ValueError(f"sqlite URL is missing a path: {url!r}")
        # urlparse gives '/relative/path.db' for sqlite:///relative/path.db.
        # We treat a leading '/' as relative to the cwd unless the next char
        # is also '/' (then it's an absolute path, sqlite:////abs/path.db).
        if path_str.startswith(("//", "/")):
            path = Path(path_str[1:])
        else:
            path = Path(path_str)
        return DatabaseConfig(kind="sqlite", raw_url=url, sqlite_path=path)
    if scheme in ("postgresql", "postgres"):
        normalised = "postgresql://" + url.split("://", 1)[1]
        return DatabaseConfig(kind="postgres", raw_url=url, postgres_dsn=normalised)
    raise ValueError(
        f"Unsupported telemetry DB scheme {scheme!r}. Use sqlite:// or postgresql://."
    )


def open_backend(url: str | None = None) -> TelemetryBackend:
    """Resolve the configured DB URL and return an initialised backend."""

    config = parse_database_url(resolve_database_url(url))
    if config.kind == "sqlite":
        from gpqa_cmab.telemetry_db.sqlite_backend import SqliteBackend

        assert config.sqlite_path is not None
        backend: TelemetryBackend = SqliteBackend(config.sqlite_path)
    else:
        from gpqa_cmab.telemetry_db.postgres_backend import PostgresBackend

        assert config.postgres_dsn is not None
        backend = PostgresBackend(config.postgres_dsn)
    backend.initialize()
    return backend


__all__ = [
    "DatabaseConfig",
    "open_backend",
    "parse_database_url",
    "resolve_database_url",
]
