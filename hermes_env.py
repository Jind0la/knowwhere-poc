"""Read Hermes profile secrets without mutating os.environ."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


@lru_cache(maxsize=1)
def _load_dotenv() -> dict[str, str]:
    """Parse HERMES_HOME/.env once; never write to os.environ."""
    env_path = _hermes_home() / ".env"
    secrets: dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        secrets[key] = value
    return secrets


def get_secret(key: str, default: str = "") -> str:
    """Return env var if set; else exact key from HERMES_HOME/.env."""
    val = os.environ.get(key)
    if val:
        return val
    return _load_dotenv().get(key, default)


def get_knowwhere_db_url() -> str:
    """DB URL: os.environ, then Hermes .env, then knowwhere_db DEFAULT."""
    url = get_secret("KNOWWHERE_DB_URL", "")
    if url:
        return url
    try:
        from knowwhere_db import DEFAULT_DB_URL

        return DEFAULT_DB_URL or ""
    except Exception:
        return ""
