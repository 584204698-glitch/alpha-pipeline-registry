from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factors import discover_factor_files, load_factor_from_path
from pipeline_utils import get_logger, read_json, safe_float


@dataclass
class BacktestArtifacts:
    factor_frame: pd.DataFrame
    results: dict[str, Any]


class BacktestEngine:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.data_path = self.project_root / "data" / "data_storage.parquet"
        self.production_path = self.project_root / "config" / "production_factors.json"
        self.main_config = read_json(self.project_root / "config" / "main_config.json", default={}) or {}
        self.transaction_fee_bps = safe_float(self.main_config.get("transaction_fee_bps", 5.0))
        self.slippage_bps = safe_float(self.main_config.get("slippage_bps", 3.0))
        self.spread_bps = safe_float(self.main_config.get("spread_bps", 1.0))
        self.total_cost_bps = self.transaction_fee_bps + self.slippage_bps + self.spread_bps
        self.logger = get_logger(self.project_root, "BACKTEST")

    def _load_data(self, min_symbols: int = 100) -> pd.DataFrame:
        override = os.environ.get("ALPHA_PIPELINE_DATA_PATH", "").strip()
        data_path = Path(override) if override else self.data_path
        data = pd.read_parquet(data_path)
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index(["timestamp", "symbol"]).sort_index()
        data = data.sort_index()
        # Keep only timestamps with enough symbols for meaningful cross-sectional IC
        symbol_counts = data.groupby(level="timestamp").size()
        dense_ts = symbol_counts[symbol_counts >= min_symbols].index
        data = data.loc[dense_ts]
        dropped = len(symbol_counts) - len(dense_ts)
        if dropped > 0:
            self.logger.info(f"Filtered {dropped} sparse timestamps, kept {len(dense_ts)} with ≥{min_symbols} symbols")
        return data

    @staticmethod
    def _forward_returns(data: pd.DataFrame, periods: int = 1) -> pd.Series:
        close = data["close"]
        grouped = close.groupby(level="symbol", group_keys=False)
        future = grouped.shift(-periods)
        returns = (future / close) - 1.0
        return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _cost_adjusted_forward_returns(
        signal: pd.Series,
        forward_returns: pd.Series,
        total_cost_bps: float,
    ) -> pd.Series:
        """Apply transaction costs: fee + slippage + spread, charged per-side on turnover."""
        cost_rate = float(total_cost_bps) / 10000.0
        direction = np.sign(signal.fillna(0.0))
        direction = pd.Series(direction, index=signal.index, dtype=float)
        position_change = direction.groupby(level="symbol", group_keys=False).diff().abs().fillna(direction.abs())
        turnover = position_change.clip(lower=0.0)
        costs = turnover * cost_rate
        adjusted = forward_returns - costs
        return adjusted.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _cross_sectional_ic(signal: pd.Series, forward_returns: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"signal": signal, "forward_return": forward_returns}).dropna()
        frame = frame.reset_index()
        daily_ic = frame.groupby("timestamp").apply(
            lambda part: part["signal"].corr(part["forward_return"], method="spearman")
            if part["symbol"].nunique() > 1
            else np.nan
        )
        return daily_ic.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _icir(signal: pd.Series, forward_returns: pd.Series) -> float:
        daily_ic = BacktestEngine._cross_sectional_ic(signal, forward_returns)
        std = daily_ic.std()
        if std is None or std == 0 or np.isnan(std):
            return 0.0
        return float(daily_ic.mean() / std)

    @staticmethod
    def _tstat(signal: pd.Series, forward_returns: pd.Series) -> float:
        daily_ic = BacktestEngine._cross_sectional_ic(signal, forward_returns)
        n = len(daily_ic)
        if n < 2:
            return 0.0
        std = daily_ic.std()
        if std is None or std == 0 or np.isnan(std):
            return 0.0
        return float(daily_ic.mean() / (std / np.sqrt(n)))

    @staticmethod
    def _quantile_returns(signal: pd.Series, forward_returns: pd.Series) -> list[float]:
        frame = pd.DataFrame({"signal": signal, "forward_return": forward_returns}).dropna().reset_index()

        def assign_quintile(part: pd.DataFrame) -> pd.DataFrame:
            unique = part["signal"].nunique()
            if unique < 5:
                ranks = part["signal"].rank(method="first", pct=True)
                part["quintile"] = np.minimum((ranks * 5).astype(int), 4)
            else:
                part["quintile"] = pd.qcut(part["signal"], 5, labels=False, duplicates="drop")
            return part

        enriched = frame.groupby("timestamp", group_keys=False).apply(assign_quintile)
        quintile_mean = enriched.groupby("quintile")["forward_return"].mean()
        return [safe_float(quintile_mean.get(i, 0.0)) for i in range(5)]

    def _regime_ic(self, signal: pd.Series, forward_returns: pd.Series, data: pd.DataFrame) -> dict[str, float]:
        returns = data["close"].groupby(level="symbol", group_keys=False).pct_change().fillna(0.0)
        realized_vol = returns.groupby(level="symbol", group_keys=False).rolling(30).std().reset_index(level=0, drop=True)
        market_vol = realized_vol.groupby(level="timestamp").mean().fillna(0.0)
        q1 = market_vol.quantile(0.33)
        q2 = market_vol.quantile(0.66)

        timestamp_regime: dict[Any, str] = {}
        for ts, value in market_vol.items():
            if value <= q1:
                timestamp_regime[ts] = "low_vol"
            elif value <= q2:
                timestamp_regime[ts] = "med_vol"
            else:
                timestamp_regime[ts] = "high_vol"

        frame = pd.DataFrame({"signal": signal, "forward_return": forward_returns}).dropna().reset_index()
        frame["regime"] = frame["timestamp"].map(timestamp_regime).fillna("med_vol")
        result: dict[str, float] = {}
        for regime in ["high_vol", "med_vol", "low_vol"]:
            subset = frame.loc[frame["regime"] == regime]
            if subset.empty or subset["symbol"].nunique() < 2:
                result[regime] = 0.0
            else:
                result[regime] = safe_float(subset["signal"].corr(subset["forward_return"], method="spearman"))
        return result

    def _active_factor_correlations(self, data: pd.DataFrame, signal: pd.Series, candidate_name: str, candidate_path: str = "") -> dict[str, dict[str, float]]:
        payload = read_json(self.production_path, default={"factors": []}) or {"factors": []}
        correlations: dict[str, dict[str, float]] = {}
        for item in payload.get("factors", []):
            factor_file = item.get("factor_file")
            factor_name = item.get("factor_name", "")
            if not factor_file:
                continue
            # Skip self — don't compare a factor against itself
            if factor_name == candidate_name or factor_file == candidate_path:
                continue
            path = self.project_root / factor_file
            if not path.exists():
                continue
            loaded = load_factor_from_path(path)
            existing_signal = loaded.factor.compute(data)
            if existing_signal.empty:
                continue
            pearson = safe_float(signal.corr(existing_signal, method="pearson"))
            spearman = safe_float(signal.corr(existing_signal, method="spearman"))
            correlations[item.get("factor_name", path.stem)] = {"pearson": pearson, "spearman": spearman}
        return correlations

    def run_backtest(self, factor_path: Path) -> dict[str, Any]:
        data = self._load_data()
        loaded = load_factor_from_path(factor_path)
        signal = loaded.factor.compute(data).reindex(data.index).fillna(0.0)
        raw_forward_returns = self._forward_returns(data)
        forward_returns = self._cost_adjusted_forward_returns(signal, raw_forward_returns, self.total_cost_bps)
        overall_ic = safe_float(signal.corr(forward_returns, method="spearman"))
        overall_ir = self._icir(signal, forward_returns)

        # Daily cross-sectional IC series (for statistical inference)
        daily_ic = self._cross_sectional_ic(signal, forward_returns)
        daily_ic_clean = daily_ic.replace([np.inf, -np.inf], np.nan).dropna()

        timestamps = sorted(data.index.get_level_values("timestamp").unique())
        split_is = timestamps[:120] if len(timestamps) > 180 else timestamps[: max(1, int(len(timestamps) * 0.66))]
        split_oos = timestamps[120:180] if len(timestamps) > 180 else timestamps[max(1, int(len(timestamps) * 0.66)) :]
        is_mask = signal.index.get_level_values("timestamp").isin(split_is)
        oos_mask = signal.index.get_level_values("timestamp").isin(split_oos)

        # OOS daily IC series
        oos_daily_ic = self._cross_sectional_ic(signal[oos_mask], forward_returns[oos_mask])
        oos_daily_ic_clean = oos_daily_ic.replace([np.inf, -np.inf], np.nan).dropna()

        quantile_returns = self._quantile_returns(signal, forward_returns)
        regime_ic = self._regime_ic(signal, forward_returns, data)
        correlations = self._active_factor_correlations(data, signal, loaded.factor.factor_name, str(factor_path))

        metrics = {
            "overall_ic": overall_ic,
            "overall_ir": overall_ir,
            "in_sample_icir": self._icir(signal[is_mask], forward_returns[is_mask]),
            "out_of_sample_icir": self._icir(signal[oos_mask], forward_returns[oos_mask]),
            "in_sample_tstat": self._tstat(signal[is_mask], forward_returns[is_mask]),
            "out_of_sample_tstat": self._tstat(signal[oos_mask], forward_returns[oos_mask]),
            "quantile_returns": quantile_returns,
            "regime_ic": regime_ic,
        }
        results = {
            "factor_name": loaded.factor.factor_name,
            "factor_file": str(Path("factors") / factor_path.name),
            "factor_class_name": loaded.class_name,
            "parameters": dict(loaded.factor.parameters),
            "required_data": loaded.factor.get_required_data(),
            "metrics": metrics,
            "correlations": correlations,
            "costs": {
                "transaction_fee_bps": self.transaction_fee_bps,
                "slippage_bps": self.slippage_bps,
                "spread_bps": self.spread_bps,
                "total_cost_bps": self.total_cost_bps,
            },
            "signal_summary": {
                "long_ratio": safe_float((signal > 0).mean()),
                "nonzero_ratio": safe_float((signal != 0).mean()),
            },
            "ic_series": {
                "daily_ic": [float(v) for v in daily_ic_clean.values],
                "oos_daily_ic": [float(v) for v in oos_daily_ic_clean.values],
                "n_daily": int(len(daily_ic_clean)),
                "n_oos": int(len(oos_daily_ic_clean)),
            },
        }
        self.logger.info(
            f"Backtested factor [{loaded.factor.factor_name}] with overall_ic={overall_ic:.4f}, overall_ir={overall_ir:.4f}"
        )
        return results

    def run_backtest_by_name(self, factor_name: str) -> dict[str, Any]:
        for path in discover_factor_files(self.project_root / "factors"):
            loaded = load_factor_from_path(path)
            if loaded.factor.factor_name == factor_name:
                return self.run_backtest(path)
        raise FileNotFoundError(f"Factor [{factor_name}] not found")
