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
    cost_input_usd_per_1m_tokens: float = 0.0
    cost_cached_input_usd_per_1m_tokens: float = 0.0
    cost_output_usd_per_1m_tokens: float = 0.0
    cost_usd_per_1k_tokens: float = 0.0
    log_level: str = "INFO"
    reasoning_effort: str | None = None
    # --- cost / call safety caps -----------------------------------------
    # Run-wide ceilings honoured by every experiment loop that calls real
    # providers. ``None`` disables that dimension. CLI flags take precedence
    # over these env-derived defaults.
    max_total_api_calls: int | None = None
    max_total_cost_usd: float | None = None
    # --- LLM client hygiene ----------------------------------------------
    # When set the OpenAI-compatible client passes ``max_tokens`` /
    # ``max_output_tokens``. Without it, reasoning models can stream tens of
    # thousands of (billed) reasoning tokens per call.
    max_output_tokens: int | None = None
    # JSON parse-and-retry attempts inside ``complete_validated``. Each retry
    # is a new billed LLM call, so the default is intentionally low.
    json_max_retries: int = 2


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
        lambda_token=_float_env("LAMBDA_TOKEN", default=0.05),
        lambda_call=_float_env("LAMBDA_CALL", default=0.01),
        cost_input_usd_per_1m_tokens=_float_env(
            "COST_INPUT_USD_PER_1M_TOKENS", default=0.0
        ),
        cost_cached_input_usd_per_1m_tokens=_float_env(
            "COST_CACHED_INPUT_USD_PER_1M_TOKENS", default=0.0
        ),
        cost_output_usd_per_1m_tokens=_float_env(
            "COST_OUTPUT_USD_PER_1M_TOKENS", default=0.0
        ),
        cost_usd_per_1k_tokens=_float_env("COST_USD_PER_1K_TOKENS", default=0.0),
        log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        reasoning_effort=(os.environ.get("REASONING_EFFORT") or "").strip().lower()
        or None,
        max_total_api_calls=_optional_int(
            os.environ.get("MAX_TOTAL_API_CALLS"), name="MAX_TOTAL_API_CALLS"
        ),
        max_total_cost_usd=_optional_float(
            os.environ.get("MAX_TOTAL_COST_USD"), name="MAX_TOTAL_COST_USD"
        ),
        max_output_tokens=_optional_int(
            os.environ.get("MAX_OUTPUT_TOKENS"), name="MAX_OUTPUT_TOKENS"
        ),
        json_max_retries=_optional_int(
            os.environ.get("LLM_JSON_MAX_RETRIES"), name="LLM_JSON_MAX_RETRIES"
        )
        or 2,
    )


def _float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}")
    return value


def _optional_int(raw: str | None, *, name: str = "value") -> int | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def _optional_float(raw: str | None, *, name: str = "value") -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}")
    return value


def clear_settings_cache() -> None:
    get_settings.cache_clear()
