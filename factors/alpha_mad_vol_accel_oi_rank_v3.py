"""
MADVolAccelOIRankV3 — OI delta=8 variant

Logic:
Same as V1 (vol_window=12, accel_lag=6, window=24) but oi_delta=8.
Slightly different OI lookback to marginally shift signal distribution.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 24) * (rank_pct(delta(open_interest, 8), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankV3(FactorRegistry):
    factor_name = "MADVolAccelOIRankV3"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 24, "oi_delta": 8}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "V1 with oi_delta=8 for marginal signal shift. Maintains vol_window=12 "
        "which is the empirically strongest setting."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 24) "
        "* (rank_pct(delta(open_interest, 8), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
