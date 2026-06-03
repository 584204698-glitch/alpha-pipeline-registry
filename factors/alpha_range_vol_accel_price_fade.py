"""
RangeVolAccelPriceFade — Range Vol Accel + Price Momentum Fade

Logic (振幅波动加速+价格反转):
VolAccelFade variant using HIGH-LOW RANGE volatility instead of
close-to-close returns. Combines range vol-accel (regime transition)
with price momentum (directional signal).

The range-based vol-accel captures the microstructure volatility component
that VolAccelFade misses — bar-level price excursions that signal imminent
regime shifts.

Formula:
-1 * zscore(delta(rolling_std((high-low)/close, 12), 6), 24) * zscore(pct_change(close, 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeVolAccelPriceFade(FactorRegistry):
    factor_name = "RangeVolAccelPriceFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "mom_lag": 6}
    inputs = ["high", "low", "close"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "VolAccelFade variant with range-based volatility. Range vol accel "
        "captures intra-bar microstructure regime shifts; price momentum provides "
        "directional fade target. Orthogonal to return-based VolAccelFade."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std((high - low) / close, 12), 6), 24) "
        "* zscore(pct_change(close, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
