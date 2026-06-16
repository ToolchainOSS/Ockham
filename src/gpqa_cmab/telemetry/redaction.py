"""Secret redaction and environment sanitization for telemetry.

Telemetry records must never persist API keys or tokens. This module knows
which env vars are sensitive (by name marker) and scrubs their values out of
prompts, responses, raw payloads, and the recorded environment snapshot.
"""

from __future__ import annotations

import os
from typing import Any

from gpqa_cmab.telemetry.hashing import _sha256_text

_RECORDED_ENV_VARS = (
    "LLM_PROVIDER",
    "MAIN_MODEL",
    "SUBAGENT_MODEL",
    "SELF_CONSISTENCY_MODEL",
    "REASONING_EFFORT",
    "LLM_USE_RESPONSES_API",
    "MAX_OUTPUT_TOKENS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_KEY_COOLDOWN_S",
    "OPENAI_MAX_RETRIES",
    "OPENAI_MAX_WAIT_S",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "LLM_DEFAULT_HEADERS",
    "LLM_TIMEOUT_S",
    "LAMBDA_TOKEN",
    "LAMBDA_CALL",
    "COST_INPUT_USD_PER_1M_TOKENS",
    "COST_CACHED_INPUT_USD_PER_1M_TOKENS",
    "COST_OUTPUT_USD_PER_1M_TOKENS",
    "MAX_TOTAL_API_CALLS",
    "MAX_TOTAL_COST_USD",
    "LLM_JSON_MAX_RETRIES",
)

_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "HEADER")


def sanitized_environment() -> dict[str, object]:
    return {
        name: _sanitize_env_value(name, os.environ.get(name))
        for name in _RECORDED_ENV_VARS
    }


def redact_known_secrets(text: str | None) -> str | None:
    if text is None:
        return None
    redacted = text
    for name in _RECORDED_ENV_VARS:
        value = os.environ.get(name)
        if not value or not _is_secret_env(name):
            continue
        for secret in _split_secret_values(value):
            if len(secret) >= 8:
                redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_known_secrets_in_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_known_secrets(value)
    if isinstance(value, list):
        return [redact_known_secrets_in_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_known_secrets_in_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): redact_known_secrets_in_value(item) for key, item in value.items()
        }
    return value


def _redact_metadata(metadata: dict[str, str]) -> dict[str, str]:
    return {key: redact_known_secrets(value) or "" for key, value in metadata.items()}


def _sanitize_env_value(name: str, value: str | None) -> dict[str, object]:
    if value is None:
        return {"set": False}
    if _is_secret_env(name):
        parts = _split_secret_values(value)
        return {
            "set": True,
            "redacted": True,
            "items": len(parts),
            "sha256": _sha256_text(value),
            "item_sha256": [_sha256_text(part) for part in parts],
            "length": len(value),
        }
    return {"set": True, "value": value}


def _is_secret_env(name: str) -> bool:
    return any(marker in name for marker in _SECRET_ENV_MARKERS)


def _split_secret_values(value: str) -> list[str]:
    parts: list[str] = []
    for chunk in value.replace("\n", ",").split(","):
        parts.extend(part for part in chunk.split() if part)
    return parts or [value]
