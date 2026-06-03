from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_CACHE: dict[tuple[str, str], logging.Logger] = {}


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        module_name = getattr(record, "module_name", record.name.upper())
        return f"[{timestamp}] [{module_name}] [{record.levelname}] - {record.getMessage()}"


def get_logger(project_root: Path, module_name: str) -> logging.Logger:
    project_root = Path(project_root)
    key = (str(project_root), module_name.upper())
    if key in _LOGGER_CACHE:
        return _LOGGER_CACHE[key]

    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    logger = logging.getLogger(f"alpha_pipeline.{module_name.lower()}.{hash(key)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(lambda record: setattr(record, "module_name", module_name.upper()) or True)
        logger.addHandler(handler)

    _LOGGER_CACHE[key] = logger
    return logger


def utc_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> Path:
    path = ensure_parent(Path(path))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:
            return default
        return result
    except Exception:
        return default
