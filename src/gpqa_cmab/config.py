from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    main_model: str = "mock-main"
    subagent_model: str = "mock-subagent"
    self_consistency_model: str = "mock-self-consistency"
    lambda_token: float = 0.05
    lambda_call: float = 0.01
    cost_usd_per_1k_tokens: float = 0.0
    log_level: str = "INFO"


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parse a single `KEY=VALUE` style line. Returns None for blanks/comments.

    Supports a leading ``export``, ignores inline comments after unquoted
    values, and strips a single matching pair of surrounding quotes.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, _, value = stripped.partition("=")
    key = key.strip()
    if not key or not key.replace("_", "").isalnum():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        # Strip an inline comment on unquoted values: ``FOO=bar  # note``.
        hash_index = value.find(" #")
        if hash_index >= 0:
            value = value[:hash_index].rstrip()
    return key, value


def _find_dotenv(start: Path | None = None) -> Path | None:
    """Search for a `.env` file from `start` upward to the filesystem root."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path | None = None, *, override: bool = False) -> Path | None:
    """Load environment variables from a `.env` file.

    By default, walks up from the current working directory to find a `.env`
    file (so commands work whether invoked from the repo root or a subdir).
    Real environment variables take precedence unless `override=True`. Safe to
    call multiple times. Returns the path that was loaded, or `None`.
    """
    dotenv_path = path if path is not None else _find_dotenv()
    if dotenv_path is None or not dotenv_path.is_file():
        return None
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw_line in text.splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return dotenv_path


@cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "mock").lower(),
        main_model=os.environ.get("MAIN_MODEL", "mock-main"),
        subagent_model=os.environ.get("SUBAGENT_MODEL", "mock-subagent"),
        self_consistency_model=os.environ.get(
            "SELF_CONSISTENCY_MODEL", "mock-self-consistency"
        ),
        lambda_token=float(os.environ.get("LAMBDA_TOKEN", "0.05")),
        lambda_call=float(os.environ.get("LAMBDA_CALL", "0.01")),
        cost_usd_per_1k_tokens=float(os.environ.get("COST_USD_PER_1K_TOKENS", "0.0")),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    )


def clear_settings_cache() -> None:
    get_settings.cache_clear()
