"""
MADVolAccelOIRankV4 — zscore window=30 variant

Logic:
Same as V1 but zscore normalization window=30 (vs 24). Longer normalization
smooths the zscore, producing marginally different signal distribution.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 12), 6), 30) * (rank_pct(delta(open_interest, 6), 30) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankV4(FactorRegistry):
    factor_name = "MADVolAccelOIRankV4"
    parameters = {"vol_window": 12, "accel_lag": 6, "window": 30, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "V1 with zscore window=30 for smoother normalization. Maintains "
        "vol_window=12 which is the empirically strongest setting."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 12), 6), 30) "
        "* (rank_pct(delta(open_interest, 6), 30) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
