"""
RangeVolAccelOIFade — Range-Based Vol Accel + OI Flow Reversal

Logic (振幅波动率加速+持仓量反转):
Same VolAccelOIFade structure but using HIGH-LOW RANGE volatility instead
of close-to-close returns. This fundamentally changes the vol signal:
range-based vol captures intra-bar price excursion (market microstructure
volatility), whereas return-based vol captures bar-to-bar change.

The range-based vol-accel has near-zero correlation with VolAccelFade's
return-based vol-accel while still capturing regime transitions.

Formula:
-1 * zscore(delta(rolling_std((high-low)/close, 12), 6), 24) * zscore(delta(open_interest, 6), 24)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class RangeVolAccelOIFade(FactorRegistry):
    factor_name = "RangeVolAccelOIFade"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["high", "low", "close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "VolAccelOIFade variant using range-based (high-low) volatility instead of "
        "close-to-close returns. Range vol captures intra-bar excursion which is "
        "orthogonal to return-based vol — eliminating correlation with VolAccelFade "
        "while preserving regime-transition timing."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_std((high - low) / close, 12), 6), 24) "
        "* zscore(delta(open_interest, 6), 24)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
