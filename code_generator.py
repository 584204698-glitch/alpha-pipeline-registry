from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent
from typing import Any

from pipeline_utils import get_logger


class FactorCodeGenerator:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.factors_dir = self.project_root / "factors"
        self.factors_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.project_root, "CODE_GENERATOR")

    @staticmethod
    def _snake_case(name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
        return cleaned.lower()

    @staticmethod
    def _class_name(name: str) -> str:
        parts = re.split(r"[^a-zA-Z0-9]+", name)
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    def render_factor_code(self, idea: dict[str, Any]) -> str:
        factor_name = idea["factor_name"]
        class_name = idea.get("class_name") or self._class_name(factor_name)
        inputs = idea.get("inputs", [])
        parameters = idea.get("parameters", {})
        timeframes = idea.get("timeframes", [])
        formula = idea.get("mathematical_formula", "0")
        rationale = idea.get("rationale", "")
        if factor_name.strip().lower() == "factorname":
            raise ValueError("DeepSeek returned placeholder factor_name 'FactorName'")
        code = f'''from __future__ import annotations

import pandas as pd

from factors.base import FactorRegistry, evaluate_formula


class {class_name}(FactorRegistry):
    factor_name = {factor_name!r}
    parameters = {parameters!r}
    inputs = {inputs!r}
    timeframes = {timeframes!r}
    rationale = {rationale!r}
    mathematical_formula = {formula!r}

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
'''
        return dedent(code)

    def generate_factor_file(self, idea: dict[str, Any]) -> Path:
        file_name = f"alpha_{self._snake_case(idea['factor_name'])}.py"
        path = self.factors_dir / file_name
        path.write_text(self.render_factor_code(idea), encoding="utf-8")
        self.logger.info(f"Generated factor file [{path.name}] for factor [{idea['factor_name']}]")
        return path
