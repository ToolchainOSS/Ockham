"""Round-robin API-key pool with 429 backoff, plus the retry-delay parsing
that drives it. This is part of the vendor boundary: the ``openai`` SDK is
imported lazily here for the ``RateLimitError`` type and client construction.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class _KeyPool:
    """Round-robin pool of OpenAI clients that share equivalent API keys.

    On a `RateLimitError` (HTTP 429) the offending key is parked for a
    delay window derived from the server's hint (``Retry-After`` header or
    "try again in Xs" message body); if no hint is provided we fall back to
    ``cooldown_seconds``. The next available key is then used.

    When *all* keys are parked (the common case for a single-key pool, or
    for multiple keys that share an org-wide TPM bucket), the pool sleeps
    until the soonest key becomes free and retries — bounded by
    ``max_attempts`` and ``max_wait_seconds``.

    Designed for the OpenAI Python SDK's ``OpenAI`` / ``AzureOpenAI`` clients
    but the type is generic; we only call ``.chat.completions.create`` and
    ``.responses.create`` on each member.
    """

    # Heuristic floor so we never sleep less than this between retries after
    # a 429 (avoids hammering the provider when its hint is essentially zero).
    _MIN_WAIT_SECONDS = 0.5

    def __init__(
        self,
        clients: list[Any],
        *,
        cooldown_seconds: float = 30.0,
        max_attempts: int = 6,
        max_wait_seconds: float = 120.0,
        rate_limit_exception: type[BaseException] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not clients:
            raise ValueError("_KeyPool requires at least one client.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if max_wait_seconds < 0:
            raise ValueError("max_wait_seconds must be >= 0.")
        self._clients = clients
        self._cooldown_seconds = cooldown_seconds
        self._max_attempts = max_attempts
        self._max_wait_seconds = max_wait_seconds
        self._cooldown_until = [0.0] * len(clients)
        self._cursor = 0
        self._lock = threading.Lock()
        self._clock = clock
        self._sleeper = sleeper
        if rate_limit_exception is None:
            rate_limit_exception = _import_rate_limit_exception()
        self._rate_limit_exception = rate_limit_exception

    @property
    def size(self) -> int:
        return len(self._clients)

    def _next_available(self) -> tuple[int, Any] | None:
        now = self._clock()
        n = len(self._clients)
        with self._lock:
            for offset in range(n):
                index = (self._cursor + offset) % n
                if self._cooldown_until[index] <= now:
                    self._cursor = (index + 1) % n
                    return index, self._clients[index]
        return None

    def _seconds_until_next_free(self) -> float:
        with self._lock:
            soonest = min(self._cooldown_until)
        return max(0.0, soonest - self._clock())

    def _park(self, index: int, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[index] = self._clock() + max(seconds, 0.0)

    def execute(self, call: Callable[[Any], Any]) -> Any:
        """Run ``call(client)`` on a free key, rotating + waiting on 429s."""
        last_error: BaseException | None = None
        attempts = 0
        deadline = self._clock() + self._max_wait_seconds

        while attempts < self._max_attempts:
            picked = self._next_available()
            if picked is not None:
                index, client = picked
                try:
                    return call(client)
                except self._rate_limit_exception as exc:
                    last_error = exc
                    attempts += 1
                    hinted = _extract_retry_delay(exc)
                    delay = hinted if hinted is not None else self._cooldown_seconds
                    delay = max(delay, self._MIN_WAIT_SECONDS)
                    self._park(index, delay)
                    logger.warning(
                        "openai_key_rate_limited index=%d delay_s=%.2f "
                        "attempt=%d/%d pool_size=%d hinted=%s",
                        index,
                        delay,
                        attempts,
                        self._max_attempts,
                        len(self._clients),
                        "yes" if hinted is not None else "no",
                    )
                    continue

            # No key is currently free. Sleep until the soonest one is, then
            # retry — provided we still have budget.
            wait_for = self._seconds_until_next_free()
            now = self._clock()
            if wait_for <= 0:
                # Race: a key just became free. Loop and pick it up.
                continue
            if now + wait_for > deadline:
                logger.error(
                    "openai_pool_exhausted reason=budget all_keys_cooling "
                    "wait_s=%.2f budget_remaining_s=%.2f pool_size=%d",
                    wait_for,
                    max(0.0, deadline - now),
                    len(self._clients),
                )
                break
            logger.info(
                "openai_pool_waiting_all_keys_cooling sleep_s=%.2f pool_size=%d",
                wait_for,
                len(self._clients),
            )
            self._sleeper(wait_for)

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"All {len(self._clients)} OpenAI API keys are cooling down."
        )


# Regex matches Groq-style messages: "Please try again in 2.4s" or
# "try again in 1m30s" / "try again in 500ms". We keep the parser narrow
# (seconds + minutes + milliseconds) since OpenAI/Groq/Together all use
# similar phrasing.
_RETRY_BODY_PATTERNS = (
    re.compile(r"try\s+again\s+in\s+([\d.]+)\s*ms", re.IGNORECASE),
    re.compile(r"try\s+again\s+in\s+(?:(\d+)\s*m)?\s*([\d.]+)\s*s", re.IGNORECASE),
)


def _extract_retry_delay(exc: BaseException) -> float | None:
    """Extract a retry delay (seconds) from a RateLimitError.

    Order of preference:
      1. ``Retry-After`` HTTP header on the response (RFC 7231).
      2. Provider-specific hint embedded in the error body
         (Groq/Together: "Please try again in X.Ys").

    Returns ``None`` if no hint is available; callers should then fall back
    to their configured cooldown.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None) or {}
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            value = None
        if value:
            try:
                parsed = float(value)
                if parsed >= 0:
                    return parsed
            except TypeError, ValueError:
                pass

    # Fall back to scraping the error body / message.
    message_sources: list[str] = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message_sources.append(err["message"])
        elif isinstance(body.get("message"), str):
            message_sources.append(body["message"])
    message_sources.append(str(exc))

    for text in message_sources:
        # ms first to avoid the seconds regex matching it as "0s" by accident.
        ms_match = _RETRY_BODY_PATTERNS[0].search(text)
        if ms_match:
            try:
                return float(ms_match.group(1)) / 1000.0
            except ValueError:
                pass
        s_match = _RETRY_BODY_PATTERNS[1].search(text)
        if s_match:
            try:
                minutes = float(s_match.group(1) or 0)
                seconds = float(s_match.group(2))
                return minutes * 60.0 + seconds
            except ValueError:
                pass
    return None


def _import_rate_limit_exception() -> type[BaseException]:
    """Import the OpenAI SDK's RateLimitError lazily.

    Falls back to a sentinel exception class that nothing will raise so the
    pool effectively disables rotation when the SDK is unavailable (tests
    typically inject a stub).
    """
    try:
        from openai import RateLimitError
    except Exception:  # pragma: no cover - openai installed in CI

        class _UnreachableRateLimit(Exception):
            pass

        return _UnreachableRateLimit
    else:
        return RateLimitError


def _build_openai_clients_for_pool(
    api_keys: list[str], shared_kwargs: dict[str, Any]
) -> list[Any]:
    """Construct one OpenAI client per key, sharing all other settings."""
    from openai import OpenAI

    return [OpenAI(**{**shared_kwargs, "api_key": key}) for key in api_keys]
