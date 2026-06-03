"""
TakerFlowMomentum — Taker Imbalance Trend-Following Factor

Logic (主动量趋势跟随):
ALL prior taker-volume factors used reversal logic (-1 * ...) — and all failed
or underperformed. Taker imbalance is NOT a reversal signal; it's a CONVICTION
signal. When aggressive buyers dominate AND price is trending, follow the trend.
The taker ratio amplifies conviction, not contrarianism.

This is the first pure taker-based MOMENTUM factor in the pipeline.

Formula:
zscore(taker_volume/(volume + 1e-8), 12) * zscore(pct_change(close, 6), 24) * rank_pct(volume, 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class TakerFlowMomentum(FactorRegistry):
    factor_name = "TakerFlowMomentum"
    parameters = {"taker_window": 12, "mom_lag": 6, "window": 24}
    inputs = ["taker_volume", "volume", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Taker imbalance is conviction, not contrarianism. Prior taker-reversal "
        "factors all failed because they faded what should have been followed. "
        "When taker buyers dominate and price is trending, follow the crowd — "
        "the taker ratio amplifies trend conviction. Multiplied by volume rank "
        "to filter low-liquidity noise."
    )
    mathematical_formula = (
        "zscore(taker_volume / (volume + 1e-8), 12) "
        "* zscore(pct_change(close, 6), 24) "
        "* rank_pct(volume, 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
