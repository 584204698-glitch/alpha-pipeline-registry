from __future__ import annotations

import os
from pathlib import Path


def load_runtime_env(env_path: str | Path | None = None) -> dict[str, str]:
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / '.env.runtime'
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded[key] = value
    return loaded
