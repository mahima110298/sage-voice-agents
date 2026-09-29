"""Load multiturn_study/.env into os.environ (only for vars not already set).

Same tiny parser as impact_speech_submission/experiment/scripts/env_utils.py
-- no python-dotenv dependency, just KEY=VALUE lines. The .env file is
gitignored; keys live only on disk, never typed into a shell command or
committed."""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value
