"""Factor ensemble: combine multiple factors with ICIR/t-stat/corr penalty weighting.

Score = Σ w_i × z(Factor_i)
where w_i = max(ICIR_i, 0) × clip(t_i/2, 0, 1) × CorrPenalty_i / Σ(...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine import BacktestEngine
from factors import load_factor_from_path
from pipeline_utils import safe_float


class FactorEnsemble:
    """Combine multiple factors into a weighted composite signal."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.engine = BacktestEngine(project_root)
        self.weights: dict[str, float] = {}
        self.factor_paths: dict[str, Path] = {}

    def add_factor(self, factor_path: Path, icir: float | None = None, tstat: float | None = None) -> None:
        """Register a factor for the ensemble."""
        loaded = load_factor_from_path(factor_path)
        name = loaded.factor.factor_name
        self.factor_paths[name] = factor_path
        if icir is not None:
            self.weights[name] = max(icir, 0.0)

    def compute_weights(
        self,
        icir_values: dict[str, float] | None = None,
        tstat_values: dict[str, float] | None = None,
        corr_matrix: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Compute ICIR × t-stat clip × correlation penalty weights.

        Args:
            icir_values: {factor_name: ICIR}
            tstat_values: {factor_name: Newey-West t-stat}
            corr_matrix: DataFrame of factor signal correlations
        """
        n = len(self.factor_paths)
        if n == 0:
            return {}

        names = list(self.factor_paths.keys())

        # Default ICIR from backtest results
        if icir_values is None:
            icir_values = {}
            for name, path in self.factor_paths.items():
                bt = self.engine.run_backtest(path)
                icir_values[name] = safe_float(bt.get("metrics", {}).get("overall_ir", 0.0))

        # Default t-stat to 1.0 if not provided
        if tstat_values is None:
            tstat_values = {name: 1.0 for name in names}

        # Default correlation matrix
        if corr_matrix is None:
            data = self.engine._load_data()
            signals = {}
            for name, path in self.factor_paths.items():
                loaded = load_factor_from_path(path)
                signals[name] = loaded.factor.compute(data).fillna(0.0)
            corr_df = pd.DataFrame({n: s for n, s in signals.items()})
            corr_matrix = corr_df.corr()

        raw_weights = {}
        for i, name in enumerate(names):
            icir_i = max(icir_values.get(name, 0.0), 0.0)
            t_i = tstat_values.get(name, 1.0)
            t_clip = min(max(t_i / 2.0, 0.0), 1.0)

            # Correlation penalty: 1 / (1 + sum of absolute correlations to others)
            corr_penalty = 1.0
            for j, other in enumerate(names):
                if i == j:
                    continue
                try:
                    abs_corr = abs(corr_matrix.loc[name, other])
                except (KeyError, AttributeError):
                    abs_corr = 0.0
                corr_penalty += abs_corr
            corr_penalty = 1.0 / corr_penalty

            raw_weights[name] = icir_i * t_clip * corr_penalty

        total = sum(raw_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in raw_weights.items()}
        else:
            self.weights = {k: 1.0 / n for k in names}

        return dict(self.weights)

    def compute_composite_signal(
        self,
        data: pd.DataFrame | None = None,
        zscore_signals: bool = True,
    ) -> pd.Series:
        """Compute weighted composite signal.

        Args:
            data: Market data DataFrame. Loads from default if None.
            zscore_signals: Whether to cross-sectional zscore each factor before combining

        Returns:
            Composite signal Series
        """
        if data is None:
            data = self.engine._load_data()

        if not self.weights or not self.factor_paths:
            raise ValueError("No factors registered. Call add_factor() and compute_weights() first.")

        composite = pd.Series(0.0, index=data.index)
        for name, path in self.factor_paths.items():
            if self.weights.get(name, 0.0) <= 0:
                continue
            loaded = load_factor_from_path(path)
            sig = loaded.factor.compute(data).reindex(data.index).fillna(0.0)

            if zscore_signals:
                # Cross-sectional zscore per timestamp
                sig_df = pd.DataFrame({"sig": sig})
                sig_df["ts"] = sig.index.get_level_values("timestamp")
                sig = sig.groupby(level="timestamp").transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                ).fillna(0.0)

            composite += self.weights[name] * sig

        return composite.fillna(0.0)


def compute_factor_correlations(
    project_root: Path,
    factor_names: list[str],
) -> pd.DataFrame:
    """Compute pairwise signal correlations between factors."""
    root = Path(project_root)
    engine = BacktestEngine(root)
    data = engine._load_data()

    signals = {}
    for fname in factor_names:
        found = None
        fname_clean = fname.lower().replace("_", "")
        for py_file in (root / "factors").glob("alpha_*.py"):
            if fname_clean in py_file.stem.lower().replace("_", ""):
                found = py_file
                break
        if found is None:
            continue
        loaded = load_factor_from_path(found)
        signals[fname] = loaded.factor.compute(data).fillna(0.0)

    df = pd.DataFrame(signals)
    return df.corr()
