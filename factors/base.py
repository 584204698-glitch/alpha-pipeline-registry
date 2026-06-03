from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {}


class FactorRegistry:
    factor_name: str = "base_factor"
    parameters: dict[str, Any] = {}
    inputs: list[str] = []
    mathematical_formula: str = ""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        self.parameters = merged

    def get_required_data(self) -> list[str]:
        raise NotImplementedError

    def compute(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


def _groupby_symbol(series: pd.Series):
    if isinstance(series.index, pd.MultiIndex) and "symbol" in series.index.names:
        return series.groupby(level="symbol", group_keys=False)
    return None


def delta(series: pd.Series, periods: int = 1) -> pd.Series:
    grouped = _groupby_symbol(series)
    return grouped.diff(periods) if grouped is not None else series.diff(periods)


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    grouped = _groupby_symbol(series)
    return grouped.pct_change(periods) if grouped is not None else series.pct_change(periods)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).mean())
    return series.rolling(window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).std())
    return series.rolling(window).std()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).max())
    return series.rolling(window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).min())
    return series.rolling(window).min()


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).sum())
    return series.rolling(window).sum()


def rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        return grouped.transform(lambda values: values.rolling(window).quantile(q))
    return series.rolling(window).quantile(q)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = rolling_mean(series, window)
    std = rolling_std(series, window).replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rank_pct(series: pd.Series, window: int) -> pd.Series:
    grouped = _groupby_symbol(series)
    if grouped is not None:
        result = grouped.rolling(window).apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)
        return result.reset_index(level=0, drop=True)
    return series.rolling(window).apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)


def indicator(condition: pd.Series | np.ndarray | list[bool]) -> pd.Series:
    if isinstance(condition, pd.Series):
        return condition.astype(float)
    return pd.Series(condition, dtype=float)


def clip(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def abs_(series: pd.Series) -> pd.Series:
    return series.abs()


def rolling_median(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 4)).median()


def rolling_mad(series: pd.Series, window: int) -> pd.Series:
    """Rolling median absolute deviation — robust to outliers."""
    roll_med = series.rolling(window, min_periods=max(2, window // 4)).median()
    return (series - roll_med).abs().rolling(window, min_periods=max(2, window // 4)).median()


def robust_zscore(series: pd.Series, window: int) -> pd.Series:
    """Robust z-score using median/MAD instead of mean/std. Handles fat-tailed crypto returns."""
    roll_med = series.rolling(window, min_periods=max(2, window // 4)).median()
    roll_mad = rolling_mad(series, window)
    return (series - roll_med) / (roll_mad + 1e-8)


def math_log(series: pd.Series) -> pd.Series:
    """Natural log — useful for normalizing volume/price ratios."""
    return np.log(series.abs() + 1e-8)


def sign(series: pd.Series) -> pd.Series:
    return np.sign(series)


SAFE_FUNCTIONS.update(
    {
        "delta": delta,
        "pct_change": pct_change,
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_median": rolling_median,
        "rolling_mad": rolling_mad,
        "rolling_max": rolling_max,
        "rolling_min": rolling_min,
        "rolling_sum": rolling_sum,
        "rolling_quantile": rolling_quantile,
        "robust_zscore": robust_zscore,
        "zscore": zscore,
        "rank_pct": rank_pct,
        "indicator": indicator,
        "clip": clip,
        "abs": abs_,
        "sign": sign,
        "log": math_log,
        "np": np,
    }
)


def evaluate_formula(formula: str, data: pd.DataFrame, parameters: dict[str, Any] | None = None) -> pd.Series:
    context: dict[str, Any] = {column: data[column] for column in data.columns}
    context.update(SAFE_FUNCTIONS)
    if parameters:
        context.update(parameters)
    result = eval(formula, {"__builtins__": {}}, context)  # noqa: S307 - restricted context
    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=data.index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class LoadedFactor:
    factor: FactorRegistry
    module_name: str
    path: Path
    class_name: str = field(default="")


def load_factor_from_path(path: Path) -> LoadedFactor:
    path = Path(path)
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load factor module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, FactorRegistry) and obj is not FactorRegistry:
            instance = obj()
            return LoadedFactor(factor=instance, module_name=module.__name__, path=path, class_name=name)
    raise ValueError(f"No FactorRegistry subclass found in {path}")


def discover_factor_files(factors_dir: Path) -> list[Path]:
    factors_dir = Path(factors_dir)
    return sorted(
        path
        for path in factors_dir.glob("*.py")
        if path.name not in {"__init__.py", "base.py"} and not path.name.startswith("_")
    )
