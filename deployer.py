from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_utils import get_logger, read_json, utc_now_z, write_json


class Deployer:
    def __init__(self, production_path: Path) -> None:
        self.production_path = Path(production_path)
        self.project_root = self.production_path.resolve().parents[1]
        self.logger = get_logger(self.project_root, "DEPLOYER")

    @staticmethod
    def _enabled_key(factor_name: str) -> str:
        return f"FACTOR_{factor_name.upper()}_ENABLED"

    def deploy(self, payload: dict[str, Any]) -> Path:
        production = read_json(self.production_path, default={"factors": []}) or {"factors": []}
        factors = production.setdefault("factors", [])
        enabled_key = self._enabled_key(payload["factor_name"])
        item = {
            "factor_name": payload["factor_name"],
            "factor_file": payload["factor_file"],
            "parameters": payload.get("parameters", {}),
            "metrics": payload.get("metrics", {}),
            "enabled_key": enabled_key,
            enabled_key: False,
            "deployed_at": utc_now_z(),
        }
        factors = [existing for existing in factors if existing.get("factor_name") != payload["factor_name"]]
        factors.append(item)
        production["factors"] = factors
        write_json(self.production_path, production)
        self.logger.info(f"Deployed factor [{payload['factor_name']}] with default disabled switch [{enabled_key}=False]")
        return self.production_path
