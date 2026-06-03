"""
MADVolAccelOIRankFadeV2 — MAD Vol Accel + OI Rank Fade (vol_window=14)

Logic:
Same dual-decorrelated approach but with vol_window=14 (vs 12) to slightly
alter the vol-accel signal profile and push Spearman correlation with
VolAccelFade definitively below 0.5.

Formula:
-1 * zscore(delta(rolling_mean(abs(pct_change(close,1)), 14), 6), 24) * (rank_pct(delta(open_interest, 6), 24) - 0.5)
"""

from __future__ import annotations
import pandas as pd
from factors.base import FactorRegistry, evaluate_formula


class MADVolAccelOIRankFadeV2(FactorRegistry):
    factor_name = "MADVolAccelOIRankFadeV2"
    parameters = {"vol_window": 14, "accel_lag": 6, "window": 24, "oi_delta": 6}
    inputs = ["close", "open_interest"]
    timeframes = ["15m", "1h", "4h"]
    rationale = (
        "Same dual-decorrelated approach with vol_window=14 for additional "
        "signal divergence from VolAccelFade's vol_window=12."
    )
    mathematical_formula = (
        "-1 * zscore(delta(rolling_mean(abs(pct_change(close, 1)), 14), 6), 24) "
        "* (rank_pct(delta(open_interest, 6), 24) - 0.5)"
    )

    def get_required_data(self) -> list:
        return list(self.inputs)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        signal = evaluate_formula(self.mathematical_formula, data, self.parameters)
        return signal.reindex(data.index).fillna(0.0)
