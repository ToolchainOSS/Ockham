"""Environment-variable parsing and validation for the OpenAI-compatible
client. Pure and vendor-neutral: nothing here imports the ``openai`` SDK.
"""

from __future__ import annotations

import json
from collections.abc import Callable


def _parse_headers(raw: str | None) -> dict[str, str] | None:
    """Parse default headers from env. Accepts JSON object or k=v,k=v list."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LLM_DEFAULT_HEADERS: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("LLM_DEFAULT_HEADERS JSON must be an object.")
        return {str(k): str(v) for k, v in value.items()}
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(
                f"Invalid LLM_DEFAULT_HEADERS entry {pair!r}; expected key=value."
            )
        key, value = pair.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers or None


def _parse_env[T](
    value: str | None,
    parser: Callable[[str], T],
    *,
    var_name: str,
    validator: Callable[[T], None] | None = None,
) -> T | None:
    """Parse an optional env string into a typed value.

    Returns ``None`` for unset / blank input. Raises ``ValueError`` (with the
    env var name) when ``parser`` or ``validator`` rejects the value.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = parser(stripped)
    except ValueError as exc:
        raise ValueError(f"Invalid {var_name}={value!r}: {exc}") from exc
    if validator is not None:
        validator(parsed)
    return parsed


def _parse_env_or[T](
    value: str | None,
    parser: Callable[[str], T],
    *,
    var_name: str,
    default: T,
    validator: Callable[[T], None] | None = None,
) -> T:
    parsed = _parse_env(value, parser, var_name=var_name, validator=validator)
    return default if parsed is None else parsed


def _require_positive(value: float, *, var_name: str) -> None:
    if value < 1:
        raise ValueError(f"{var_name} must be >= 1 (got {value}).")


def _require_nonneg(value: float, *, var_name: str) -> None:
    if value < 0:
        raise ValueError(f"{var_name} must be >= 0 (got {value}).")


_VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _resolve_timeout(value: str | None) -> float | None:
    return _parse_env(value, float, var_name="LLM_TIMEOUT_S")


def _resolve_reasoning_effort(value: str | None) -> str | None:
    """Normalize and validate a reasoning effort value.

    Modern OpenAI reasoning models (gpt-5.x, o-series) plus several
    OpenAI-compatible providers accept reasoning effort with values from the
    set: ``none | minimal | low | medium | high | xhigh``. Some models only
    accept a subset; the API will reject unsupported values at request time.
    Empty / None disables reasoning configuration entirely.
    """
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if stripped not in _VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid REASONING_EFFORT={value!r}. "
            f"Expected one of: {sorted(_VALID_REASONING_EFFORTS)}."
        )
    return stripped


def _resolve_max_output_tokens(value: str | None) -> int | None:
    return _parse_env(
        value,
        int,
        var_name="MAX_OUTPUT_TOKENS",
        validator=lambda v: _require_positive(v, var_name="MAX_OUTPUT_TOKENS"),
    )


def _resolve_api_keys(
    explicit_single: str | None,
    explicit_multi: str | None,
    *,
    env_keys: str | None,
    env_legacy_key: str | None,
    env_alt_key: str | None,
) -> list[str]:
    """Build the ordered list of API keys to load-balance across.

    Precedence (first non-empty wins):
      1. Explicit constructor `api_keys=[...]` (handled by caller via
         `explicit_multi` as a pre-comma-joined string).
      2. Explicit constructor `api_key="..."` (single key).
      3. `OPENAI_API_KEYS` env (comma- or whitespace-separated).
      4. `OPENAI_API_KEY` env (single).
      5. `LLM_API_KEY` env (single, used by many self-hosted servers).
      6. Sentinel "not-needed" for keyless local endpoints.

    All keys are assumed to be EQUIVALENT (same model access, same org).
    Duplicates are removed while preserving order.
    """
    candidates: list[str] = []
    if explicit_multi:
        candidates = _split_keys(explicit_multi)
    elif explicit_single:
        candidates = [explicit_single]
    elif env_keys:
        candidates = _split_keys(env_keys)
    elif env_legacy_key:
        candidates = [env_legacy_key]
    elif env_alt_key:
        candidates = [env_alt_key]
    else:
        candidates = ["not-needed"]
    # Dedupe while preserving order; drop empties.
    seen: set[str] = set()
    unique: list[str] = []
    for raw_key in candidates:
        key = raw_key.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    if not unique:
        unique = ["not-needed"]
    return unique


def _split_keys(raw: str) -> list[str]:
    """Split a multi-key string by comma, newline, or whitespace."""
    parts: list[str] = []
    for raw_chunk in raw.replace("\n", ",").split(","):
        chunk = raw_chunk.strip()
        if chunk:
            parts.extend(part for part in chunk.split() if part)
    return parts


def _resolve_cooldown_seconds(value: str | None, *, default: float = 30.0) -> float:
    return _parse_env_or(
        value,
        float,
        var_name="OPENAI_KEY_COOLDOWN_S",
        default=default,
        validator=lambda v: _require_nonneg(v, var_name="OPENAI_KEY_COOLDOWN_S"),
    )


def _resolve_positive_int(value: str | None, *, default: int, var_name: str) -> int:
    return _parse_env_or(
        value,
        int,
        var_name=var_name,
        default=default,
        validator=lambda v: _require_positive(v, var_name=var_name),
    )


def _resolve_nonneg_float(value: str | None, *, default: float, var_name: str) -> float:
    return _parse_env_or(
        value,
        float,
        var_name=var_name,
        default=default,
        validator=lambda v: _require_nonneg(v, var_name=var_name),
    )
