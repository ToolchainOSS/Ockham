from __future__ import annotations

from functools import cache
from hashlib import sha256
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    return sha256(load_prompt(name).encode("utf-8")).hexdigest()


def prompt_version(name: str) -> str:
    return f"{name}:{prompt_hash(name)[:12]}"
