"""
MADVolAccelOIRankV7 — accel_lag=5, oi_delta=8

Logic:
V3 base with accel_lag reduced from 6 to 5. Slightly faster vol-accel
detection to shift signal profile while keeping vol_window=12.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 5), 24) * (rank_pct(delta(open_interest, 8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankV7(FactorRegistry):
    factor_name = "MADVolAccelOIRankV7"
    parameters = {"vol_window": 12, "accel_lag": 5, "window": 24, "oi_delta": 8}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = "V3 with accel_lag=5 for faster vol-accel detection."
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 5), 24) "
        "* (rank_pct(delta(open_interest, 8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
