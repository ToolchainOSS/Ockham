from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache


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


@cache
def get_settings() -> Settings:
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
